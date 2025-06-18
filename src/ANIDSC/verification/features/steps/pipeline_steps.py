from behave import given, when, then
from ANIDSC.verification.toy_components.feature_extractor import BaseTrafficFeatureExtractor
from ANIDSC.verification.toy_components.feature_buffer    import BaseFeatureBuffer
from ANIDSC.verification.toy_components.detection         import BaseDetector
from ANIDSC.verification.toy_components.pipeline_component import Pipeline

# Map the names in your .feature to the actual classes
COMPONENT_MAP = {
    "BaseTrafficFeatureExtractor": BaseTrafficFeatureExtractor,
    "BaseFeatureBuffer":           BaseFeatureBuffer,
    "BaseDetector":                BaseDetector,
}

@given("a fresh pipeline builder")
def step_fresh_builder(context):
    context.pipeline = None

@given('a component "{name}" with output schema')
def step_output_schema(context, name):
    # Instantiate the component and store its schema if you want to validate it
    context.last_component   = COMPONENT_MAP[name]()
    context.expected_schema  = [(row['field'], row['type']) for row in context.table]

@given('a component "{name}" with input schema')
def step_input_schema(context, name):
    # Instantiate the component and store its schema if you want to validate it
    context.next_component   = COMPONENT_MAP[name]()
    context.expected_schema  = [(row['field'], row['type']) for row in context.table]

@when("I assemble the pipeline components in order")
def step_assemble(context):
    # Read the list of component names from the table
    names = [row[0] for row in context.table]
    # Save for the Then steps
    context.component_names = names

    # Instantiate and chain them
    comps    = [COMPONENT_MAP[n]() for n in names]
    pipeline = comps[0]
    for c in comps[1:]:
        pipeline = pipeline | c
    context.pipeline = pipeline

@then("the pipeline should be valid")
def step_assert_valid(context):
    # 1) Must be a Pipeline instance
    assert isinstance(context.pipeline, Pipeline), f"Expected Pipeline, got {context.pipeline!r}"

    # 2) Must have exactly as many stages as in the table
    expected_count = len(context.component_names)
    actual_count   = len(context.pipeline.components)
    assert actual_count == expected_count, f"Expected {expected_count} components, got {actual_count}"

    # 3) Must preserve the order of components
    actual_names = [c.__class__.__name__ for c in context.pipeline.components]
    assert actual_names == context.component_names, (
        f"Order mismatch: expected {context.component_names!r}, got {actual_names!r}"
    )

@then('the pipeline validation should fail with message')
def step_assert_fail(context):
    # Re-use the same list of names from the When step
    names = context.component_names
    comps = [COMPONENT_MAP[n]() for n in names]
    try:
        p = comps[0]
        for c in comps[1:]:
            p = p | c
        assert False, "Expected a TypeError but none was raised"
    except TypeError as e:
        # The triple-quoted block under the step is in context.text
        expected = context.text.strip()
        actual   = str(e)
        assert expected in actual, (
            f"Expected error to contain:\n{expected}\nbut got:\n{actual}"
        )
