classdef ConvLayer < handle
    %ConvLayer NN layer that does only convolution
    
    properties
        outputChannels
        % Store weights as a cell array of weight cell arrays
        channelWeights
        channelBiases
        
        lastForward
    end
    
    methods
        function obj = ConvLayer(fullInputWidth, fullInputHeight, ...
                numOutputChannels, numInputChannels, kernelSideLength, ...
                stride)
            
            % weightInitEpsilon is a weight initialization hyperparameter
            weightInitEpsilon = 0.1;
            
            %TODO: note assumption of things like max kernel and square
            %kernel and 2k+1 side length
            obj.channelWeights = cell(1, numOutputChannels);
            % Biases can just be stored as a vector, since only 1 per
            % channel
            
            obj.channelBiases = rand(1, numOutputChannels);
            obj.channelBiases = obj.channelBiases * 2;
            obj.channelBiases = obj.channelBiases - 1;
            obj.channelBiases = obj.channelBiases * weightInitEpsilon;
            
            obj.outputChannels = numOutputChannels;
            
            for channelNum = 1:numOutputChannels
                % rand function gives random uniform values for the
                % specified dimensions.
                
                % Weight cell array dimensions for each layer are square
                % with side length kernelSideLength and then quantity of
                % cells equal to the number of input channels
                
                W = cell(1, numInputChannels);
                
                for inCh = 1:numInputChannels
                    % kernelSideLength x kernelSideLength kernel size
                    W{inCh} = rand(kernelSideLength);
                    % Weights are small random values
                    W{inCh} = W{inCh} * 2;
                    W{inCh} = W{inCh} - 1;
                    W{inCh} = W{inCh} * weightInitEpsilon;
                end
                
                obj.channelWeights{channelNum} = W;
            end
        end
        
        function out = forward(obj, inputCellArray)
            % At least for now, just do default zero-padding with MATLAB'S
            % conv2 function
            
            % Current understanding of convolutional layers: one output
            % matrix per channel
            
            obj.lastForward = cell(1, obj.outputChannels);
            % For every output channel
            for outCh = 1:obj.outputChannels 
                % For every input channel
                curTotal = zeros(size(inputCellArray{1}));
                for inCh = 1:size(inputCellArray,2)
                    curTotal = curTotal + conv2(inputCellArray{inCh}, ...
                        obj.channelWeights{outCh}{inCh}, 'same');
                end
                obj.lastForward{outCh} = curTotal + ...
                    obj.channelBiases(outCh);
            end
            out = obj.lastForward;
          
        end
        
        function weightedSensitivitiesCellArray = backward(obj, ...
                nextLayerSensitivities)
            weightedSensitivitiesCellArray = ...
                cell(size(nextLayerSensitivities));
            % I believe the algorithm for sensitivity propagation is just
            % convolution with kernel rot180
            for ch = 1:size(nextLayerSensitivities,2)
                curConvKernel = rot90(rot90(obj.channelWeights{ch}));
                weightedSensitivitiesCellArray{ch} = ...
                    conv2(nextLayerSensitivities{ch}, curConvKernel);
            end
            
            % Now, update weights for current convolutional layer
            for ch = 1:size(obj.channelWeights, 2)
                % Weights must be updated through convolution as well?
            end
        end
    end
end

