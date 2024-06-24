function [v,d] = listId2matId(i,order)
    v = fix(i/order) + 1;
    d = mod(i,order);
    if d == 0
        v = i/order;
        d = order;
    end
end