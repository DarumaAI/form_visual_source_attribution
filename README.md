# form_visual_source_attribution
Simple post-generation visual source-attribution (developed for form-type documents).
This python module is built to test multimodal encoders on visual context-attribution tasks.


# What is visual context-attribution?
Visual context attribution consists in linking parts of the output of a LLM to parts of the context.
In this case, this simple implementation is used to link an entire answer to an entire image.
You can chunk your images as you wish and use the package anyways.


# What is to be expected in the future?
I am currently working on a fine-grained visual context-attributor that performs multiple object detections on multiple document images.
Also, this fine grained attributor will be LLM-agnostic and quite fast. Stay in tune for datasets and models.
