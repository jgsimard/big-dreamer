from unittest import TestCase


class TestFreezeParameters(TestCase):

    def test_freeze_parameters(self):
        from utils import FreezeParameters
        import torchvision.models as models

        m = models.resnet18()

        for p in m.parameters():
            self.assertTrue(p.requires_grad)

        with FreezeParameters([m]):
            for p in m.parameters():
                self.assertFalse(p.requires_grad)

        for p in m.parameters():
            self.assertTrue(p.requires_grad)
