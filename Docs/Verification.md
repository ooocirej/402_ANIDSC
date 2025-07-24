# Verification
There is currently an initial implementation of the verification that can be found in src/ANIDSC/verification

# Toy_Components
A toy implementation of the pipeline can be found on

verification/toy_components/pipeline_component.py

that mimics the actual pipeline logic of the NIDS. The __or__ function of PipelineComponent class checks the input and output of the components.
This checks the abstract method "process" which is implemented within the toy components in the 

verification/toy_components folder

# Behave 
../verification/features 
Contains a Behave test that uses the toy implementation of the pipeline and components to check whether the verification works as expected.
The steps implementation can be found in 
../verification/features/steps
and run using

behave ../verification/features