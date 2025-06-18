from behave import given, when, then
from ANIDSC.verification.toy_components.feature_extractor import BaseTrafficFeatureExtractor
from ANIDSC.verification.toy_components.feature_buffer    import BaseFeatureBuffer
from ANIDSC.verification.toy_components.detection         import BaseDetector
from ANIDSC.verification.toy_components.pipeline_component import Pipeline

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
    context.last_component = COMPONENT_MAP[name]()
    context.expected_schema = [(row['field'], row['type']) for row in context.table]

@given('a component "{name}" with input schema')
def step_input_schema(context, name):
    context.next_component = COMPONENT_MAP[name]()
    context.expected_schema = [(row['field'], row['type']) for row in context.table]

@when("I assemble the pipeline components in order")
def step_assemble(context):
    # Read the list of component names from the table
    names = [row[0] for row in context.table]
    context.component_names = names

    # Instantiate and chain
    comps = [COMPONENT_MAP[n]() for n in names]
    pipeline = comps[0]
    for c in comps[1:]:
        pipeline = pipeline | c
    context.pipeline = pipeline

@then("the pipeline should be valid")
def step_assert_valid(context):
    assert isinstance(context.pipeline, Pipeline), f"Expected Pipeline, got {context.pipeline!r}"

    expected_count = len(context.component_names)
    actual_count   = len(context.pipeline.components)
    assert actual_count == expected_count, f"Expected {expected_count} components, got {actual_count}"

    actual_names = [c.__class__.__name__ for c in context.pipeline.components]
    assert actual_names == context.component_names, (
        f"Order mismatch: expected {context.component_names!r}, got {actual_names!r}"
    )

@then('the pipeline validation should fail with message')
def step_assert_fail(context):
    # Re-use the same component_names from the @when
    names = context.component_names
    comps = [COMPONENT_MAP[n]() for n in names]
    print(comps)

    try:
        p = comps[0]
        for c in comps[1:]:
            p = p | c
        assert False, "Expected a TypeError but none was raised"
    except TypeError as e:
        expected = context.text.strip()
        actual   = str(e)
        assert expected in actual, f"Expected error to contain:\n{expected}\nbut got:\n{actual}"
