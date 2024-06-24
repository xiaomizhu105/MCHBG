clear all; clc;
%%
% load dataset
dataname='HW';
load(strcat(dataname,'.mat'));
n_view = length(X);
c = length(unique(Y));  
n = size(X{1},1);


% 数据集标准化
for v = 1:n_view
    for  j = 1:n
        X{v}(j,:) = ( X{v}(j,:) - mean( X{v}(j,:) ) ) / std( X{v}(j,:) );
    end
end
% for v = 1:n_view
% %     XX = X{v};
% %     for n = 1:size(XX,1)
%     a = max(X{v}(:));
%     X{v} = double(X{v}./a);
% %     XX(n,:)=XX(n,:)./norm(XX(n,:),'fro');
% %     end
% %     X{v} = double(XX);
% end

%%
% setting
rng(0);% 
anchor_rate = 0.3; % 
opts.style = 4; % 
k = 10; % 
order = 4; % 
M = n_view + 1; %
r = 1;

IterMax = 50;
m = fix(n*anchor_rate);

%% 锚点采样
B = cell(n_view,1); % 高阶二部图存储元胞
centers = cell(n_view,1); % 锚点特征矩阵存储元胞
disp('----------Anchor Selection----------');
if opts.style == 1 % direct sample
    XX = [];
    for v = 1:length(X)
       XX = [XX X{v}];
    end
    [~,ind,~] = graphgen_anchor(XX,m);
    for v = 1:n_view
        centers{v} = X{v}(ind, :);
    end
elseif opts. style == 2 % rand sample
    vec = randperm(n);
    ind = vec(1:m);
    for v = 1:n_view
        centers{v} = X{v}(ind, :);
    end
elseif opts. style == 3 % KNP
    XX = [];
    for v = 1:n_view
        XX = [XX X{v}];
    end
    [~, ~, ~, ~, dis] = litekmeans(XX, m);
    [~,ind] = min(dis,[],1);
    ind = sort(ind,'ascend');
    for v = 1:n_view
        centers{v} = X{v}(ind, :);
    end
elseif opts. style == 4 % kmeans sample
    XX = [];
    for v = 1:n_view
       XX = [XX X{v}];
       len(v) = size(X{v},2);
    end
    [~, Cen, ~, ~, ~] = litekmeans(XX, m);
    t1 = 1;
    for v=1:n_view
       t2 = t1+len(v)-1;
       centers{v} = Cen(:,t1:t2);
       t1 = t2+1;
    end
end

%% 各个视角一阶二部图初始化
disp('----------First order Graphs Inilization----------');
for v = 1:n_view
    D = L2_distance_1(X{v}', centers{v}');
    [~, idx] = sort(D, 2); % sort each row
    B{v} = zeros(n,m);
    for ii = 1:n
        id = idx(ii,1:k+1);
        di = D(ii, id);
        B{v}(ii,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end
end

%% 各个视角高阶二部图生成
disp('----------Generate high order 2P Graphs----------');
for v = 1:n_view
    B{v,1} = B{v,1}./max(max(B{v,1},[],2));
    [U,sigma,Vt] = svd(B{v,1});
    for d = 2:order
        temp = U*sigma.^(2*d-1)*Vt';
        temp(temp<eps)=0;
        temp = temp./max(max(temp,[],2));
        %temp = temp./sum(temp,2);
        B{v,d} = temp;
    end
end

%%
disp('----------Optimization----------');
% initial P
P = zeros(n,m);
for v = 1:n_view
    for d = 1:order
        P = P + B{v,d};
    end
end
P = P/(n_view*order);
% initial A
A = zeros(n_view,order);
a = reshape(A,[n_view*order,1]);

fx = zeros(n_view*order,1);% n_view*order x 1
obj_list = [];
for iter = 1:IterMax
    fprintf('The %d-th iteration...\n',iter);
    
    % || Fix P, update A ||
    for v = 1:n_view
        for d = 1:order
            id = (v-1)*order + d;
            fx(id) = norm(P - B{v,d}, 'fro');
        end
    end
    [~,Id] = sort(fx,'ascend');
    a = zeros(n_view*order,1);
    a(Id(1:M)) = 1;

    % || update d ||
    for i = 1:n_view*order
        [v,d] = listId2matId(i,order);
        gd(i) = r/2*a(i)*(norm(P - B{v,d}, 'fro'))^(r-2);
    end

    % || Fix A, update P ||
    G = zeros(n,m);
    for i = 1:n_view*order
        [v,d] = listId2matId(i,order);
        G = G + gd(i)*B{v,d};
    end
    G = G/sum(gd);
    [y1,~,P,~,~,~] = coclustering_bipartite_fast1(G, c, IterMax); 

    % compute obj
    obj = 0;
    for i = 1:n_view*order
        [v,d] = listId2matId(i,order);
        obj = obj + a(i)*(norm(P - B{v,d}, 'fro'))^r;
    end

    % 迭代终止条件
    obj_list = [obj_list obj];
    if iter>2 && norm(obj_list(end)-obj_list(end-1)) < 0.0000001
        break;
    end
end

ComputeAcc(P,Y);
function ComputeAcc(P,Y)
    [n, m] = size(P);
    S=sparse(n+m,n+m);
    S(1:n,n+1:end)=P; 
    S(n+1:end,1:n)=P';
    G = graph(S); 
    y = conncomp(G); 
    y1=y(1:n)';
    result = ClusteringMeasure1(Y, y1);
    fprintf('acc = %f !!\n',result(1));
end



