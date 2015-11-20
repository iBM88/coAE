
close all;
clear all;

%% settings
file_rect = 'rect_balanced_preprocessed.mat';
folder_data = 'data/';

seed = 1;

hid = 100;

% it makes sense to have different learning rates for supervised and
% unsupervised networks because they are minimizing different error
% surfaces anyway.

p_rate = 0.01;
p_momentum = 0.0;
p_decay = 0.0;
%p_noise = 0.1;
p_iter = 100;
p_batch = 100;
p_func = 'sigm'; p_func_d = 'sigm_d';
p_arch = [100 100];

%% initialize

% features: angle, ratio (standardized in 0 and 1)
load(fullfile(folder_data,file_rect));
dim = size(train_x,1);
var = size(train_y,1);
samples = size(train_x,2);
%p_batch = samples+1;

rng(seed);

w_lim_low  = - 1 / sqrt(dim);
w_lim_high = - w_lim_low;

supNet.wIn  = unifrnd(w_lim_low,w_lim_high,dim+1,hid);
supNet.wOut = unifrnd(w_lim_low,w_lim_high,hid+1,var);
supNet.wInUp= zeros(dim+1,hid);
supNet.wOutUp= zeros(hid+1,var);

unsNet.wIn  = supNet.wIn;
unsNet.wOut = unifrnd(w_lim_low,w_lim_high,hid+1,dim);
unsNet.wInUp= zeros(dim+1,hid);
unsNet.wOutUp= zeros(hid+1,dim);

errors = [];

%% train networks
for it = 1:p_iter
    for b = 0:ceil(samples/p_batch)-1
        p_start = b*p_batch+1;
        p_end   = (b+1)*p_batch;
        if(p_end > samples)
            p_end = samples;
        end
        
        %fprintf('training iteration %d batch %d\n',it,b+1);
        
        data = train_x(:,p_start:p_end);
        label = train_y(:,p_start:p_end);
        
        % supervised training
        
        supNet.z1 = [data;ones(1,size(data,2))]' * supNet.wIn;
        supNet.h  = func(supNet.z1,p_func);
        
        supNet.z2 = [supNet.h ones(size(supNet.h,1),1)] * supNet.wOut;
        supNet.o  = func(supNet.z2,p_func);
        
        supNet.e = supNet.o' - label;
        
        
        wInUp_temp = supNet.wInUp;
        wOutUp_temp = supNet.wOutUp;
        
        
        supNet.dOut = supNet.e' .* func(supNet.z2,p_func_d);
        supNet.wOutUp = ([supNet.h ones(size(supNet.h,1),1)])' * supNet.dOut;
        
        supNet.dIn  = supNet.dOut * supNet.wOut';
        supNet.wInUp  = [data;ones(1,size(data,2))] * (supNet.dIn(:,1:end-1) .* func(supNet.z1,p_func_d));
        
        supNet.wOut = supNet.wOut ...
                    - p_rate * supNet.wOutUp ...
                    + p_decay * norm(supNet.wOut,1) ...
                    + p_momentum * wOutUp_temp;
                
        supNet.wIn = supNet.wIn ...
                    - p_rate * supNet.wInUp ...
                    + p_decay * norm(supNet.wIn,1) ...
                    + p_momentum * wInUp_temp;

        % unsupervised training
        
        unsNet.z1 = [data;ones(1,size(data,2))]' * unsNet.wIn;
        unsNet.h  = func(unsNet.z1,p_func);
        
        unsNet.z2 = [unsNet.h ones(size(unsNet.h,1),1)] * unsNet.wOut;
        unsNet.o  = func(unsNet.z2,p_func);
        
        unsNet.e = unsNet.o' - data;
        
        wInUp_temp = unsNet.wInUp;
        wOutUp_temp = unsNet.wOutUp;
        
        unsNet.dOut = unsNet.e' .* func(unsNet.z2,p_func_d);
        unsNet.wOutUp = ([unsNet.h ones(size(unsNet.h,1),1)])' * unsNet.dOut;
        
        
        unsNet.dIn  = unsNet.dOut * unsNet.wOut';
        unsNet.wInUp  = [data;ones(1,size(data,2))] * (unsNet.dIn(:,1:end-1) .* func(unsNet.z1,p_func_d));
        
        unsNet.wOut = unsNet.wOut ...
                    - p_rate * unsNet.wOutUp ...
                    + p_decay * norm(unsNet.wOut,1) ...
                    + p_momentum * wOutUp_temp;
                
        unsNet.wIn = unsNet.wIn ...
                    - p_rate * unsNet.wInUp ...
                    + p_decay * norm(unsNet.wIn,1) ...
                    + p_momentum * wInUp_temp;
    end % end batch
    
    % calculate error
    sup_er_train =  calc_error( train_x, train_y ,supNet.wIn, supNet.wOut ,p_func, p_func);
    sup_er_test =  calc_error( test_x, test_y ,supNet.wIn, supNet.wOut ,p_func, p_func);
    
    uns_er_train =  calc_error( train_x, train_x ,unsNet.wIn, unsNet.wOut ,p_func, p_func);
    uns_er_test =  calc_error( test_x, test_x ,unsNet.wIn, unsNet.wOut ,p_func, p_func);
    
    errors(it,:) = [sup_er_train sup_er_test uns_er_train uns_er_test];
    
    fprintf('iteration %d: %.5f %.5f %.5f %.5f\n',it,sup_er_train,sup_er_test,uns_er_train,uns_er_test);
    
end % end iteration

figure('name','Supervised error');
plot(errors(:,1:2));
figure('name','Supervised filters');
visualize(supNet.wIn(1:end-1,:),[min(supNet.wIn(:)) max(supNet.wIn(:))],38,38);
figure('name','Supervised filter weights rotation');
bar(0.5+supNet.wOut(:,1));
figure('name','Supervised filter weights ratio');
bar(0.5+supNet.wOut(:,2));

figure('name','Unupervised error');
plot(errors(:,3:4));
figure('name','Unupervised filters');
visualize(unsNet.wIn(1:end-1,:),[min(supNet.wIn(:)) max(supNet.wIn(:))],38,38);


