function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValLoss
persistent valLag

% Clear the variables when training starts.
if info.State == "start"
    bestValLoss = 1e+19;
    valLag = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if info.ValidationLoss < bestValLoss
        disp(sprintf('valLag = %d/%d - %8.5g, %8.5g',valLag,N,bestValLoss, info.ValidationLoss))
        valLag = 0;
        bestValLoss = info.ValidationLoss;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
        disp(sprintf('valLag = %d/%d - %8.5g, %8.5g',valLag,N,bestValLoss, info.ValidationLoss))
    end
    %disp(sprintf('valLag = %d/%d - %8.5g, %8.5g',valLag,N,bestValLoss, info.ValidationLoss))
    
end

end