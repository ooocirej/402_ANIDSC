Feature: Saved and loaded torch models are equivalent

  Scenario Outline: Model round-trip equivalence
      Given a detection model component <model_name> with input_dims <ndim> on cpu
      And a fixed random input batch of size 5 and width <ndim>
      And I briefly train the component for 5 steps
      When I save the component to a temporary directory
      And I load the component back from that directory
      Then the loaded component has the same parameters as the original
      # And the loaded component produces the same outputs as the original on the fixed input

      Examples:
        | model_name         | ndim |
        | torch_model.AE     | 8    |
        | torch_model.VAE    | 8    |
        | torch_model.ICL    | 8    |
        | torch_model.Kitsune| 8    |
        | torch_model.GOAD   | 8    |
        | torch_model.SLAD   | 8    |