from ANIDSC.pipeline import Pipeline
from ANIDSC.templates import get_template
import yaml

# Step 1: Feature Extraction (saves features, NO results)
template = get_template(
    "feature_extraction",
    dataset_name="test_data",
    file_name="malicious_Port_Scanning",
    fe_class="AfterImage",
    save_buffer=True
)
pipeline = Pipeline.load(template)
pipeline.setup()
save_dir = pipeline.start()
print(f"Features extracted to: {save_dir}")

# Step 2: Detection (uses extracted features, PRODUCES results)
# Get fe_attrs from the saved feature extraction
with open(f"{save_dir}/pipeline_config.yaml") as f:
    manifest = yaml.safe_load(f)
fe_attrs = manifest["feature_extractor"]["attrs"]

template = get_template(
    "detection",
    dataset_name="test_data",
    file_name="malicious_Port_Scanning",  # CSV from feature extraction
    model_name="torch_model.AE",
    fe_name="AfterImage",
    fe_attrs=fe_attrs
)
pipeline = Pipeline.load(template)
pipeline.setup()
save_dir = pipeline.start()
print(f"Results saved to: test_data/AfterImage/results/malicious_Port_scanning/")