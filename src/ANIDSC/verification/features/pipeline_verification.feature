Feature: Pipeline component verification
  As a user of the ANIDSC verification framework
  I want to ensure that only components with matching input/output schemas can be chained
  So that I can catch schema mismatches during pipeline assembly

  Background:
    Given a fresh pipeline builder

  Scenario: Chaining compatible components succeeds
    Given a component "BaseTrafficFeatureExtractor" with output schema:
        | field     | type                          |
        | features  | Tuple[List[float], List[Any]] |
    And a component "BaseFeatureBuffer" with input schema
        | field     | type          |
        | features  | Tuple[List[float], List[Any]] |
    And a component "BaseDetector" with input schema
        | field     | type          |
        | array     | NDArray[Any]  |
    When I assemble the pipeline components in order
        | BaseTrafficFeatureExtractor |
        | BaseFeatureBuffer           |
        | BaseDetector                |
    Then the pipeline should be valid

  Scenario: Chaining extractor directly to detector fails
    Given a component "BaseTrafficFeatureExtractor" with output schema:
        | field     | type                          |
        | features  | Tuple[List[float], List[Any]] |
    And a component "BaseDetector" with input schema
        | field     | type         |
        | array     | NDArray[Any] |
    When I assemble the pipeline components in order
        | BaseTrafficFeatureExtractor   |
        | BaseDetector                  |
    Then the pipeline validation should fail with message:
        """
        Incompatible chaining: BaseTrafficFeatureExtractor returns Tuple[List[float], List[Any]], but BaseDetector expects NDArray[Any]
        """
