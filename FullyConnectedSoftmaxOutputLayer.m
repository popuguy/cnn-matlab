classdef FullyConnectedSoftmaxOutputLayer < handle
    % FullyConnectedLayer processes multiple channels only using softmax to
    % produce a single channel output
    
    properties
        learningRate
        lastForward
        
        weightMatrix
        
        lastInputFlat
        
        numInputChannels
        channelSideLength
    end
    
    methods
        function obj = FullyConnectedSoftmaxOutputLayer(learningRate, ...
                inDim, outDim, numInputChannels, channelSideLength)
            % Inputs:
            %   learningRate: step size
            %
            %   inDim: dimensionality of input vectors
            %
            %   outDim: dimensionality of output vectors
            weightInitEpsilon = 0.1;
            
            obj.numInputChannels = numInputChannels;
            obj.channelSideLength = channelSideLength;
            
%             obj.transFn = transFn;
%             obj.dTransFn = dTransFn;
            obj.learningRate = learningRate;
            
%             obj.weightMatrix = cell(1, size(layerDims, 2) - 1);
            
            % Weight matrix dimensions for each layer are (output,input)
            % In addition, the "input" dimension is expanded by 1 to
            % incorporate the bias.
            obj.weightMatrix = rand(outDim, inDim + 1);
            % At the moment, make the weights in each layer small
            % random values from -0.2 to 0.2
            obj.weightMatrix = obj.weightMatrix * 2;
            obj.weightMatrix = obj.weightMatrix - 1;
            obj.weightMatrix = obj.weightMatrix * weightInitEpsilon;

%             obj.layerWeights{layerNum} = weightMatrix;

                
                
        end
        function out = forward(obj, inputCellArray)
            % First thing's first, flatten all channels into a single
            % vector
            inputFlat = [];
            for ch = 1:size(inputCellArray, 2)
                % Append flattened columns one after another vertically
                
                % This is not an efficient way to do this, but should be
                % effective
                inputFlat = [inputFlat ; inputCellArray{ch}(:)];
            end
            
            obj.lastInputFlat = inputFlat;
            
            netOut = obj.weightMatrix * [inputFlat ; 1];
            obj.lastForward = softmax(netOut);
            
            
            out = obj.lastForward;
        end
        function weightedSensitivitiesCellArray = backward(obj, target)
            % For softmax and cross-entropy, computation is incredibly easy
            
            sensitivities = target - obj.lastForward;
            
            % Calculate with pre-update weight matrix that doesn't include 
            % the biases a value to pass back to previous layer
            weightedSensitivities = ...
                obj.weightMatrix(:,1:end-1)' * sensitivities;
            
            weightedSensitivitiesCellArray = cell(1, obj.numInputChannels);
            
            numValsPerChannel = obj.channelSideLength ^ 2;
            for ch = 1:obj.numInputChannels
                
                curStartIndex = 1 + numValsPerChannel * (ch - 1);
                weightedSensitivitiesCellArray{ch} = ...
                    reshape(weightedSensitivities(...
                    curStartIndex: ...
                    curStartIndex + numValsPerChannel - 1, 1), ...
                    obj.channelSideLength, obj.channelSideLength);
            end
            
            % Now reshape and place back into channels as the input arrived
            
            
            % Now need to update weights based on sensitivities
            % Update with -alpha sens * (prev layer a)'
            
            % Incorporate bias with a trailing '1' input
            obj.weightMatrix = obj.weightMatrix - ...
                obj.learningRate * sensitivities * [obj.lastInputFlat; 1]';
            
        end
    end
end