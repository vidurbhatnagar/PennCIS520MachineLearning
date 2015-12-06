function models = init_model()

addpath('./liblinear');
addpath('./libsvm');

modelStruct = load('models.mat');
models = modelStruct.models;