"""GPU sufficient-statistic fits for daily conditional quantiles, not indices."""
import numpy as np
import torch
from climx_sota import qmap

class Curvature:
    @staticmethod
    def raw(x):
        return np.column_stack((x[:,1]**2,x[:,1]*x[:,2],x[:,2]**2))

    def fit(self,x):
        self.projection=np.linalg.lstsq(x,self.raw(x),rcond=None)[0]
        residual=self.raw(x)-x@self.projection
        self.mean=residual.mean(0)
        self.scale=residual.std(0).clip(min=1e-10)
        return self

    def transform(self,x):
        z=(self.raw(x)-x@self.projection-self.mean)/self.scale
        return np.column_stack((x,z))

def window_quantiles(z,device):
    z=torch.as_tensor(z,dtype=torch.float32,device=device)
    win=torch.as_tensor(qmap.WIN,dtype=torch.long,device=device)
    values=z[win].sort(dim=1).values
    pos=qmap.PROBS*(values.shape[1]-1)
    lo=np.floor(pos).astype(int)
    hi=np.minimum(lo+1,values.shape[1]-1)
    fraction=torch.as_tensor((pos-lo).astype(np.float32)[None,:,None],device=device)
    return values[:,lo]*(1-fraction)+values[:,hi]*fraction

class PairedFit:
    def __init__(self,x,penalty,device):
        self.device=device
        self.p=len(penalty)
        self.curve=Curvature().fit(x)
        self.x=torch.as_tensor(self.curve.transform(x),dtype=torch.float64,device=device)
        self.penalty=torch.as_tensor(list(penalty)+[100.]*3,dtype=torch.float64,device=device)
        self.xty=None

    def add(self,i,q):
        q=q.flatten().to(dtype=torch.float64,device=self.device)
        if self.xty is None:
            self.xty=torch.zeros((self.x.shape[1],len(q)),dtype=torch.float64,device=self.device)
            self.shape=(12,len(qmap.PROBS),len(q)//(12*len(qmap.PROBS)))
        self.xty.add_(self.x[i,:,None]*q[None,:])

    def finish(self):
        a=self.x.T@self.x
        base=torch.linalg.solve(a[:self.p,:self.p]+torch.diag(self.penalty[:self.p]),self.xty[:self.p])
        curve=torch.linalg.solve(a+torch.diag(self.penalty),self.xty)
        self.b0=base.reshape((self.p,)+self.shape).float()
        self.b1=curve.reshape((self.x.shape[1],)+self.shape).float()
        del self.xty,self.x
        return self

    def daily_knots(self,x,candidate):
        row=self.curve.transform(np.asarray(x)[None,:])[0] if candidate else x
        b=self.b1 if candidate else self.b0
        q=torch.einsum('p,pmkc->mkc',torch.as_tensor(row,dtype=torch.float32,device=self.device),b).sort(dim=1).values
        q=q+torch.arange(len(qmap.PROBS),device=self.device,dtype=torch.float32)[None,:,None]*1e-6
        blend=torch.as_tensor(qmap.BLEND,dtype=torch.float32,device=self.device)
        return (blend@q.reshape(12,-1)).reshape((365,len(qmap.PROBS),self.shape[-1]))

def daily_loss(knots,truth,doy0):
    d=torch.as_tensor(doy0,dtype=torch.long,device=knots.device)
    z=torch.as_tensor(truth,dtype=torch.float32,device=knots.device)
    delta=z[:,None,:]-knots[d]
    p=torch.as_tensor(qmap.PROBS,dtype=torch.float32,device=knots.device)
    loss=torch.maximum(p[None,:,None]*delta,(p-1)[None,:,None]*delta)
    return 2*torch.trapezoid(loss,p,dim=1).double().mean(0)

def fixtures(device):
    rng=np.random.default_rng(510011)
    x=np.column_stack((np.ones(21),rng.normal(size=(21,4))))
    c=Curvature().fit(x)
    assert np.max(np.abs(x.T@(c.transform(x)[:,5:])))<1e-10
    np.testing.assert_allclose(c.transform(x)[:,5:].mean(0),0,atol=1e-12)
    np.testing.assert_allclose(c.transform(x)[:,5:].std(0),1,atol=1e-12)
    penalty=np.array([0.,1.,1.,30.,30.])
    pair=PairedFit(x,penalty,device)
    ref=qmap.QuantileModel(5,7)
    refq=qmap.QuantileModel(8,7)
    for i in range(len(x)):
        z=rng.normal(size=(365,7)).astype(np.float32)
        a=qmap.year_window_quantiles(z)
        b=window_quantiles(z,device)
        np.testing.assert_allclose(a,b.cpu().numpy(),rtol=0,atol=2e-6)
        pair.add(i,b)
        ref.add_year(x[i],a)
        refq.add_year(c.transform(x[i:i+1])[0],a)
    ref.fit(penalty);refq.fit(np.r_[penalty,[100.]*3]);pair.finish()
    np.testing.assert_allclose(pair.b0.cpu().numpy(),ref.B,rtol=1e-5,atol=2e-6)
    np.testing.assert_allclose(pair.b1.cpu().numpy(),refq.B,rtol=1e-5,atol=2e-6)
    for candidate,model in ((False,ref),(True,refq)):
        row=c.transform(x[:1])[0] if candidate else x[0]
        expected=model.daily_knots(model.knots(row,False))
        actual=pair.daily_knots(x[0],candidate)
        np.testing.assert_allclose(actual.cpu().numpy(),expected,rtol=1e-5,atol=3e-6)
        truth=rng.normal(size=(365,7)).astype(np.float32)
        loss=daily_loss(actual,truth,np.arange(365)).cpu().numpy()
        delta=truth[:,None,:]-expected
        p=qmap.PROBS[None,:,None]
        expected_loss=2*np.trapezoid(np.maximum(p*delta,(p-1)*delta),qmap.PROBS,axis=1).mean(0)
        np.testing.assert_allclose(loss,expected_loss,rtol=1e-5,atol=2e-6)
        assert np.isfinite(loss).all() and (loss>=0).all()
    return {'quantiles':'PASS','ridge_control':'PASS','ridge_Q2':'PASS','daily_interpolation':'PASS','pinball_integration':'PASS','train_only_basis':'PASS','device':str(device)}
