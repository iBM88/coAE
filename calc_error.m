function [ e ] = calc_error( data, target ,w1, w2 ,p_func1, p_func2)

    z  = [data;ones(1,size(data,2))]' * w1;
    h  = func(z,p_func1);
     
    z2 = [h ones(size(h,1),1)] * w2;
    o  = func(z2,p_func2);
    
    e = mean(0.5*mean((o' - target).^2));   % I used mean to be comparable
end

