import unittest
import numpy as np,torch
from hetst.simulator import synthetic,simulate,Inventory
from hetst.data import Prepared
from hetst.models import Model,RelAttention
from hetst.evaluation import events

torch.set_num_threads(1)
class Integrity(unittest.TestCase):
 def setUp(self):self.dem=synthetic('seasonal',55,T=120)
 def test_mass_and_nonnegative(self):
  s=simulate(self.dem);self.assertLess(max(abs(l['mass_residual']) for l in s['ledger']),1e-7);self.assertGreaterEqual(s['b'].min(),-1e-8)
 def test_causal_future_mutation(self):
  a=simulate(self.dem);d=self.dem.copy();d[100:]*=10;b=simulate(d)
  np.testing.assert_array_equal(a['x'][:100],b['x'][:100]);np.testing.assert_array_equal(a['edge'][:100],b['edge'][:100])
 def test_split_and_normalization(self):
  a=Prepared(simulate(self.dem));dem=self.dem.copy();dem[96:]*=20;b=Prepared(simulate(dem))
  np.testing.assert_array_equal(a.mean,b.mean);np.testing.assert_array_equal(a.x[:96],b.x[:96])
  self.assertLess((a.parts['train']+2).max(),72);self.assertGreaterEqual((a.parts['val']+2).min(),72);self.assertGreaterEqual(a.parts['test'].min(),96)
 def test_shift_prefix_same(self):
  a=simulate(self.dem);b=simulate(self.dem,shift='lead');np.testing.assert_array_equal(a['x'][:96],b['x'][:96])
 def test_event_matching_bounded(self):
  y=np.zeros((12,1));y[8:10]=1;p=np.zeros_like(y);p[1]=.9
  m=events(y,p);self.assertEqual(m['event_recall'],0);self.assertTrue(np.isnan(m['ewlt']))
  p[6]=.9;m=events(y,p);self.assertEqual(m['event_recall'],1);self.assertEqual(m['ewlt'],2);self.assertEqual(m['alert_count'],2)
 def test_no_repeated_backlog_events(self):
  y=np.zeros((10,1));y[4:]=1;p=np.zeros_like(y);p[2:]=.9;m=events(y,p);self.assertEqual(m['event_count'],1);self.assertEqual(m['alert_count'],1)
 def test_attention_normalization_gradient(self):
  d=Prepared(simulate(self.dem));x,e,y,b=d.batch(d.parts['train'][:3]);m=Model('HetST');a,q=m(x,e,*d.graph);(a.mean()+q.mean()).backward()
  self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()));s=m.attention.sum(-1);torch.testing.assert_close(s,torch.ones_like(s));self.assertTrue((q>=0).all())
 def test_no_edge_ablation(self):
  d=Prepared(simulate(self.dem));x,e,_,_=d.batch(d.parts['train'][:2]);m=Model('NoEdge').eval()
  a=m(x,e,*d.graph)[0];b=m(x,e+100,*d.graph)[0];torch.testing.assert_close(a,b)
 def test_single_task_regression_head_untrained(self):
  from hetst.experiment import loss
  loss.wp=loss.wn=1.;d=Prepared(simulate(self.dem));x,e,y,b=d.batch(d.parts['train'][:2]);m=Model('ClsOnly');a,q=m(x,e,*d.graph);loss(a,q,y,b,'ClsOnly').backward()
  self.assertTrue(all(p.grad is None for p in m.reg.parameters()))
 def test_emergency_mass_balance(self):
  s=simulate(self.dem);env=Inventory(s['net'],120,0)
  for row in self.dem:
   _,_,_,l=env.step(row,np.ones(env.net.n));self.assertLess(abs(l['mass_residual']),1e-7)
 def test_graph_features_are_causal_and_shape_stable(self):
  a=Prepared(simulate(self.dem));changed=self.dem.copy();changed[100:]*=20;b=Prepared(simulate(changed))
  times=a.parts['train'][:5];ga=a.graph_tabular(times);gb=b.graph_tabular(times)
  np.testing.assert_array_equal(ga,gb);self.assertEqual(ga.shape[0],len(times)*a.net.n)
  self.assertGreater(ga.shape[1],a.tabular(times).shape[1])

if __name__=='__main__':unittest.main()
