classdef ReLULayer < handle
    %ReLULayer NN layer that does only ReLU
    
    properties
%         lastNetOutput
    end
    
    methods
        function obj = ReLULayer()
        end
        function out = forward(obj, inputCellArray)
            % Since ReLU is in itself essentially a transfer function, the
            % net output is the input
%             obj.lastNetOutput = inputCellArray;
            
            % ReLU simply computes element-wise max(0, A) during forward
            % pass
            out = cell(size(inputCellArray));
            for ch = 1:size(inputCellArray,2)
                out{ch} = max(0, inputCellArray{ch});
            end
        end
        function weightedSensitivitiesCellArray = backward(obj, ...
                nextLayerSensitivities)
            % For ReLU, there are no parameters, so no update is necessary.
            % Simply pass back the sensitivities
            weightedSensitivitiesCellArray = cell(size(...
                nextLayerSensitivities));
            % First compute weighted sensitivities to backpropagate
            for ch=1:size(nextLayerSensitivities, 2)
                weightedSensitivitiesCellArray{ch} = ...
                    nextLayerSensitivities{ch} >= 0;
            end
        end
    end
end