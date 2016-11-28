local mnist = require 'mnist';
local mnist = require 'mnist';

function load_model()
	-- load the model
	model = torch.load('network.model')
	-- print ('network model loaded')
	return model.testErr
end










