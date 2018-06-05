function x = bar2x(center, N)
    x=[];
    for i = 1 : numel(center)
       x = [x; ones(round(N(i)),1).*center(i)];
    end
end

