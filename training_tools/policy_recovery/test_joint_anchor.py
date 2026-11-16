import unittest
import numpy as np
from joint_anchor import anchor_joint_chunk


class JointAnchorTests(unittest.TestCase):
    def setUp(self):
        self.state = np.zeros(16, dtype=np.float32)
        self.raw = np.zeros((50,16), dtype=np.float32)
        # Positive reconstruction error masks an intended negative motion.
        self.raw[:,0] = .08 - np.arange(50)*.001
        self.raw[:,14] = np.linspace(.95,.1,50)

    def test_preserves_intended_motion_and_discrete_io(self):
        result = anchor_joint_chunk(self.raw,self.state)
        self.assertLess(result[1,0]-self.state[0],0)
        np.testing.assert_allclose(np.diff(result[:,:14],axis=0),np.diff(self.raw[:,:14],axis=0),atol=1e-7)
        np.testing.assert_array_equal(result[0,:14],self.state[:14])
        np.testing.assert_array_equal(result[:,14:],self.raw[:,14:])
        self.assertAlmostEqual(float(self.raw[0,0]),.08,places=6)

    def test_uses_each_chunks_own_observation(self):
        first = anchor_joint_chunk(self.raw,self.state)
        later = self.state.copy(); later[0]=.02
        second = anchor_joint_chunk(self.raw,later)
        np.testing.assert_allclose(second[:,0]-first[:,0],.02,atol=1e-7)

    def test_fails_on_unbounded_or_nonfinite_bias(self):
        self.raw[0,0]=.16
        with self.assertRaises(ValueError):anchor_joint_chunk(self.raw,self.state)
        self.raw[0,0]=np.nan
        with self.assertRaises(ValueError):anchor_joint_chunk(self.raw,self.state)

    def test_identity_when_first_action_is_measured_state(self):
        self.state[:14]=self.raw[0,:14]
        np.testing.assert_allclose(anchor_joint_chunk(self.raw,self.state),self.raw,atol=1e-7)


if __name__=='__main__':unittest.main()
