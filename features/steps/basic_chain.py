from ANIDSC.pipeline import Pipeline
from ANIDSC.templates import get_template
from behave import given, when, then
from pathlib import Path


import os
import yaml



@given("a new basic pipeline with input from csv file initialized with dataset test_data, file {file}, feature extractor {fe_name} and model {model}")
def step_given_new_csv_afterimage_model(context, file, fe_name, model):

    # Read fe_attrs directly from feature extraction pipeline config
    fe_config_file = Path("test_data") / fe_name / file / "feature_extraction" / "pipeline_config.yaml"

    if not fe_config_file.exists():
        raise FileNotFoundError(
            f"Feature extraction pipeline not found at {fe_config_file}.\n"
            f"Run Feature 1 first: behave features/1_feature_extraction_chain.feature"
        )

    # Load the manifest
    with open(fe_config_file) as f:
        fe_manifest = yaml.safe_load(f)

    # Extract fe_attrs directly from the manifest
    fe_attrs = fe_manifest["feature_extractor"]["attrs"]

    # Create detection pipeline
    template = get_template(
        "detection", 
        dataset_name="test_data", 
        file_name=file, 
        model_name=model, 
        fe_name=fe_name, 
        fe_attrs=fe_attrs
    )

    context.pipeline = Pipeline.load(template)
    context.pipeline.setup()
    
    
@given("a loaded basic pipeline with input from csv file initialized with dataset test_data, file {file}, feature extractor {fe_name} and model {model}")
def step_given_loaded_csv_afterimage_model(context, file, fe_name, model):
    # if model=="BoxPlot":
    #     saved_file=f"test_data/{fe_name}/saved_components/pipeline/benign_lenovo_bulb/CSVReader->LivePercentile->{model}->BaseEvaluator.yaml"
    
    # else:
    #     saved_file=f"test_data/{fe_name}/saved_components/pipeline/benign_lenovo_bulb/CSVReader->LivePercentile->OnlineOD({model})->BaseEvaluator.yaml"
    
    # with open(saved_file) as f:
    #     manifest = yaml.safe_load(f)
        
    # manifest["attrs"]["manifest"]["data_source"]["attrs"]["file_name"]=file
    
    # context.pipeline=Pipeline.load(manifest)
    # context.pipeline.on_load()

    # NEW: Load from detection-specific directory
    save_dir = Path("test_data") / fe_name / "benign_lenovo_bulb" / "detection"
    
    if not (save_dir / "pipeline_config.yaml").exists():
        raise FileNotFoundError(
            f"No saved detection pipeline found at {save_dir}.\n"
            f"Run 'new' scenarios first to create the checkpoint."
        )
    
    # Load the saved detection pipeline
    pipeline = Pipeline.load_state(save_dir)
    
    # Update file_name for different test files
    pipeline.manifest["data_source"]["attrs"]["file_name"] = file
    pipeline.components["data_source"].file_name = file
    
    context.pipeline = pipeline
    



@then("the pipeline should not fail")
def step_then_data_processed_correctly(context):
    # the pipeline should run
    assert context.failed is False


@then("the components are saved")
def step_then_components_are_saved(context):
    # manifest_path=context.pipeline.get_save_path()
    
    # loaded_pipeline=Pipeline.load(manifest_path)

    # loaded_pipeline.on_load() # load the components
    
    # assert loaded_pipeline==context.pipeline

    # use load_state
    loaded = Pipeline.load_state(context.save_dir)
    assert set(loaded.components.keys()) == set(context.pipeline.components.keys())

    
    

