"""
Tests for the explainable AI module in Neurenix.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import neurenix
    from neurenix.explainable import ShapExplainer, LimeExplainer, FeatureImportance
    from neurenix.explainable import PartialDependence, Counterfactual, ActivationVisualization
except ImportError as e:
    print(f"Error importing neurenix: {e}")
    raise


class TestExplainableAI(unittest.TestCase):
    """Test cases for the explainable AI module."""

    def setUp(self):
        """Set up test fixtures."""
        self.X = np.random.rand(100, 10).astype(np.float32)
        self.y = np.random.randint(0, 2, 100).astype(np.float32)
        
        self.weights = np.random.rand(10).astype(np.float32)
        self.bias = np.random.rand(1).astype(np.float32)[0]
        
        self.model_fn = lambda x: np.dot(x, self.weights) + self.bias

    def test_shap_explainer(self):
        """Test SHAP explainer."""
        try:
            explainer = ShapExplainer(self.model_fn, self.X[:10])
            
            explanation = explainer.explain(self.X[0:1])
            
            self.assertEqual(len(explanation['shap_values']), 10)
            self.assertIsInstance(explanation['base_value'], float)
        except Exception as e:
            self.fail(f"ShapExplainer test failed with error: {e}")

    def test_lime_explainer(self):
        """Test LIME explainer."""
        try:
            feature_names = [f"feature_{i}" for i in range(10)]
            explainer = LimeExplainer(self.model_fn, feature_names=feature_names)
            
            explanation = explainer.explain(self.X[0])
            
            self.assertGreater(len(explanation['feature_weights']), 0)
            self.assertIsInstance(explanation['intercept'], float)
        except Exception as e:
            self.fail(f"LimeExplainer test failed with error: {e}")

    def test_feature_importance(self):
        """Test feature importance."""
        try:
            explainer = FeatureImportance(self.model_fn, self.X, self.y)
            
            explanation = explainer.explain()
            
            self.assertEqual(len(explanation['importances']), 10)
        except Exception as e:
            self.fail(f"FeatureImportance test failed with error: {e}")

    def test_partial_dependence(self):
        """Test partial dependence."""
        try:
            explainer = PartialDependence(self.model_fn, self.X, features=[0, 1])
            
            explanation = explainer.explain()
            
            self.assertIn('grid_points', explanation)
            self.assertIn('values', explanation)
        except Exception as e:
            self.fail(f"PartialDependence test failed with error: {e}")

    def test_counterfactual(self):
        """Test counterfactual."""
        try:
            explainer = Counterfactual(self.model_fn, self.X[0])
            
            explanation = explainer.explain()
            
            self.assertEqual(len(explanation['counterfactual']), 10)
            self.assertIsInstance(explanation['counterfactual_prediction'], float)
        except Exception as e:
            self.fail(f"Counterfactual test failed with error: {e}")

    def test_activation_visualization(self):
        """Test activation visualization."""
        try:
            class SimpleModel:
                def __init__(self):
                    self.layers = {
                        'layer1': np.random.rand(10, 10),
                        'layer2': np.random.rand(10, 5),
                        'layer3': np.random.rand(5, 1)
                    }
                
                def __call__(self, x):
                    activations = {}
                    activations['layer1'] = np.tanh(np.dot(x, self.layers['layer1']))
                    activations['layer2'] = np.tanh(np.dot(activations['layer1'], self.layers['layer2']))
                    activations['layer3'] = np.tanh(np.dot(activations['layer2'], self.layers['layer3']))
                    self.activations = activations
                    return activations['layer3']
                
                def get_activations(self):
                    return self.activations
            
            model = SimpleModel()
            
            explainer = ActivationVisualization(model)
            
            _ = model(self.X[0])  # Forward pass to generate activations
            explanation = explainer.explain()
            
            self.assertIn('layer1', explanation)
            self.assertIn('layer2', explanation)
            self.assertIn('layer3', explanation)
        except Exception as e:
            self.fail(f"ActivationVisualization test failed with error: {e}")


if __name__ == '__main__':
    unittest.main()
