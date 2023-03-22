function loss = gaussianNegativeLogLikelihood(predictedOutput, trueLabels)
    % Split the predicted output into means and standard deviations
    numParameters = size(predictedOutput, 2) / 2;
    predictedMeans = predictedOutput(:, 1:numParameters);
    predictedStds = predictedOutput(:, numParameters+1:end);

    % Calculate the Gaussian negative log-likelihood
    negLogLikelihood = 0.5 * (log(2 * pi * predictedStds.^2) + ((trueLabels - predictedMeans).^2) ./ (predictedStds.^2));
    loss = mean(negLogLikelihood, 'all');
end
