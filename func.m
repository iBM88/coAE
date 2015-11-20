function [ y ] = func( x ,fname)
    switch fname
        case 'sigm'
            y = 1 ./ (1 + exp(-x));
        case 'tanh'
            y = 1.7159*tanh(2/3.*x);
        case 'softplus'
            y = log(1 + exp(x));
            
            
        case 'sigm_d'
            tmp = exp(x);
            y = tmp ./ (1 + tmp).^2;
        case 'tanh_d'
            y = 1.14393 * sech(0.666667.*x).^2;
        case 'softplus_d'
            y = 1 ./ (1 + exp(-x));
    end
    y(isnan(y))=0;
%     if isnan(y)
%         x
%         assert(false,strcat('nan ' , fname));
%         y = zeros(size(x));
%     end
%     if isinf(y)
%         x
%         assert(false,strcat('inf ',fname));
%         y = zeros(size(x));
%     end
end

