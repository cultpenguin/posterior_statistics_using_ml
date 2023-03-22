classdef CustomRegressionResponseDatastore < matlab.io.Datastore & matlab.io.datastore.mixin.CustomRead
    properties
        UnderlyingDatastore
    end

    methods
        function this = CustomRegressionResponseDatastore(datastore)
            this.UnderlyingDatastore = datastore;
        end

        function data = readData(this)
            data = read(this.UnderlyingDatastore);
            data.Response = [data.Response, data.Response];
        end

        function tf = hasdata(this)
            tf = hasdata(this.UnderlyingDatastore);
        end

        function reset(this)
            reset(this.UnderlyingDatastore);
        end
    end
end
