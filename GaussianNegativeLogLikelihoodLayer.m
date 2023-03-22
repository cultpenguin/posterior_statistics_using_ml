classdef GaussianNegativeLogLikelihoodLayer < nnet.layer.RegressionLayer
    methods
        function this = GaussianNegativeLogLikelihoodLayer(name)
            % Set the layer name
            this.Name = name;

            % Set the layer description
            this.Description = "Gaussian Negative Log-Likelihood Layer";
        end

        function loss = forwardLoss(this, predictedOutput, trueLabels)
            % Call the gaussianNegativeLogLikelihood function
            loss = gaussianNegativeLogLikelihood(predictedOutput, trueLabels);
        end
    end
end