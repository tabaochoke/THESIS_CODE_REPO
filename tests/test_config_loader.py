import unittest
import os
from src.config_loader import ConfigLoader

class TestConfigLoaderDeepLearning(unittest.TestCase):
    def setUp(self):
        # Đường dẫn đến file cấu hình deep learning
        self.config_path = os.path.join(os.path.dirname(__file__), "../tests/deep_learning_config.yaml")

    def test_load_config(self):
        """Kiểm tra việc load file config"""
        config = ConfigLoader.load_config(self.config_path)
        self.assertIsNotNone(config, "Config should not be None after loading")
        self.assertIn("model", config, "Config should contain 'model' section")
        self.assertIn("training", config, "Config should contain 'training' section")

    def test_get_model_params(self):
        """Kiểm tra truy xuất tham số model"""
        ConfigLoader.load_config(self.config_path)
        input_size = ConfigLoader.get("model.input_size")
        hidden_layers = ConfigLoader.get("model.hidden_layers")
        self.assertEqual(input_size, 128, "Input size should be 128 as defined in the config")
        self.assertEqual(hidden_layers, [256, 128, 64], "Hidden layers should match the config")

    def test_get_training_params(self):
        """Kiểm tra truy xuất tham số training"""
        ConfigLoader.load_config(self.config_path)
        learning_rate = ConfigLoader.get("training.learning_rate")
        batch_size = ConfigLoader.get("training.batch_size")
        self.assertEqual(learning_rate, 0.001, "Learning rate should be 0.001")
        self.assertEqual(batch_size, 32, "Batch size should be 32")

    def test_get_dataset_params(self):
        """Kiểm tra truy xuất tham số dataset"""
        ConfigLoader.load_config(self.config_path)
        dataset_path = ConfigLoader.get("dataset.path")
        train_split = ConfigLoader.get("dataset.train_split")
        self.assertEqual(dataset_path, "/data/dataset", "Dataset path should match the config")
        self.assertAlmostEqual(train_split, 0.8, "Train split should be 0.8")

    def test_get_non_existing_key(self):
        """Kiểm tra truy cập tham số không tồn tại"""
        ConfigLoader.load_config(self.config_path)
        non_existing = ConfigLoader.get("non.existing.key", default="default_value")
        self.assertEqual(non_existing, "default_value", "Should return the default value for non-existing keys")

if __name__ == "__main__":
    unittest.main()
