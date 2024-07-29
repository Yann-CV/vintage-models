Add description here


* [ ] All new vintage model is described in a paper_review.md and is used in an experiment script
* [ ] All the new modules are tested, and these tests contain a gpu optional test
* [ ] All the new modules are compatible with the `.to(device)` instruction
* [ ] All modules are documented with a docstring
* [ ] `import torch` is used only if it is the only option (acceptable in tests or experiments)
* [ ] The loss computation are not included in the model class itself
* [ ] If the model is used to generate images, the generation function is not in the model class
