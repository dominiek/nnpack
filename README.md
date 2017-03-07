_Heavy work in progress_

# NNPack

NNPack contains tools & standards for packaging and distributing Neural Networks. Today's Deep Learning frameworks are very powerful, but they their tools for persisting and distributing Machine Learning models are lacking. The NNPack specification provides a high-level structure for packaging up various types of models and their training data (Model Scaffolds).

* [NNPack Specification 1.0-DRAFT](./SPEC.md)

## Features

* Coming soon: Easily push/pull models from the open [nnpack.org](http://nnpack.org) repository
* Developer-friendly packaging of models. Model meta-data lives in `/nnpackage.json`
* Packaging and distributing training data using Model Scaffolds
* Rich contextual meta-data for model, training data and labels

## Coming Soon: Installing a Model

```bash
nnpack pull dominiek/scene_type_vision
```

## Packaging a Model

In order to package a model, simply create an `nnpackage.json` in its root directory. All neural network state information needs to be moved into a `state/` subfolder. Here's an example:

```json
{
  "id": "58745c9fbd17c82dd4ff7c9c",
  "name": "Scene Types",
  "description": "Classify images into Indoor VS Outdoor scenes",
  "version": 0.1,
  "engines": {
    "tensorflow": ">=0.11"
  },
  "nodes": {
    "softmaxOutput": {
      "type": "tensor",
      "id": "retrained_layer:0"
    },
    "convolutionalRepresentation": {
      "type": "tensor",
      "id": "pool_3/_reshape:0"
    }
  },
  "author": {
    "name": "Dominiek Ter Heide",
    "email": "info@dominiek.com"
  },
  "labelsDefinitionFile": "labels.json",
  "stateDir": "state"
}
```

For each softmax output of the neural net, a `labels.json` provides context for what the predictions mean. Each `node_id` corresponds to a softmax output:

```json
{
  "labels": [
    {"id": "58745e65bd17c82ec1545a64", "name": "Indoor", "node_id": 0},
    {"id": "58745e69bd17c82ec1545a65", "name": "Outdoor", "node_id": 1}
  ]
}
```

In the above two files a lot of information is contained on how to run the model, what engine is required, what the outputs mean and how transfer learning can occur.

[Read Documentation](./SPEC.md)

## Coming Soon: Distributing a Model

```bash
nnpack init .
nnpack push dominiek/scene_type_vision
```

## Todo

* Encapsulate utility functions in PIP module
* Add utility functions for creating Models and Scaffolds
* Create a Spec roadmap
* Create simple website skeleton
* Create NNPack Registry beta
