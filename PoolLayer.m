classdef PoolLayer < handle
    %PoolLayer NN layer that does only max pooling
    
    properties
        lastForward
        
        poolSide
        
        % Can think about maxMask as the "weight" matrix
        maxMask
    end
    
    methods
        function obj = PoolLayer(poolingSideLength, stride)
            obj.poolSide = poolingSideLength;
        end
        function out = forward(obj, inputCellArray)
            
            obj.maxMask = cell(size(inputCellArray));
            
            
            out = cell(size(inputCellArray));
            for ch = 1:size(inputCellArray, 2)
                % Make a new mask
                obj.maxMask{ch} = zeros(size(inputCellArray{ch}));
                
                % Perform pooling on each channel of input
                out{ch} = obj.poolize(inputCellArray{ch}, ch);
            end
        end
        function pool = poolize(obj, inputMatrix, channelToUpdate)
            pool = ones(floor(size(inputMatrix, 1) / obj.poolSide), ...
                floor(size(inputMatrix, 2) / obj.poolSide)) * -inf;
            % Row and column to update in pool
            updRow = 0;
%             updCol = 0;
            for rowNum = 1 : obj.poolSide : size(inputMatrix, 1) - (obj.poolSide - 1)
                updRow = updRow + 1;
                updCol = 0;
                for colNum = 1 : obj.poolSide: size(inputMatrix, 2) - (obj.poolSide - 1)
                    updCol = updCol + 1;
                    
                    % Default at -1 so error if somehow no maximum
                        maxRow = -1;
                        maxCol = -1;
                    
                    for pRow = 0 : obj.poolSide - 1
                        
                        for pCol = 0 : obj.poolSide - 1
                            if inputMatrix(rowNum + pRow, colNum + pCol)...
                                    > pool(updRow, updCol)
                                maxRow = rowNum + pRow;
                                maxCol = colNum + pCol;
                                pool(updRow, updCol) = inputMatrix(...
                                    maxRow, maxCol);
                                
                            end
%                             pool(updRow, updCol) = max(pool(updRow, updCol), ...
%                                 inputMatrix(rowNum + pRow, ...
%                                 colNum + pCol));
                        end
                        
                        
                    end
                    % As is the derivative of the max pool
                    obj.maxMask{channelToUpdate}(maxRow, maxCol) = 1;
%                     disp("pool update");
                end
            end
            
        end
        function weightedSensitivitiesCellArray = backward(obj, ...
                nextLayerSensitivities)
            weightedSensitivitiesCellArray = cell(...
                size(nextLayerSensitivities));
            for ch=1:size(weightedSensitivitiesCellArray,2)
                weightedSensitivitiesCellArray{ch} = obj.maxMask{ch} .* ...
                    nextLayerSensitivities{ch};
            end
        end
    end
end