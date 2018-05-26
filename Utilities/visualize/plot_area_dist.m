load areas
t = datetime(time./1000,'ConvertFrom','posixTime','TimeZone','America/New_York','Format','dd-MMM-yyyy HH:mm:ss.SSS');
t = t - t(1);
nbins = 20;

selected_time = {minutes(315), minutes(135), minutes(15)};
%selected_time = flip(selected_time);
ws = 10;
working_on = beta_dist;
for i = 1 : numel(selected_time)
    index =  find(t >selected_time{i}, 1);
    cur_dist = double(working_on{index})*1.3*1.3;
    [N, edges, bins] = histcounts(cur_dist, logspace(log10(min(cur_dist)), log10(max(cur_dist)), nbins));
    center = ((edges(1:end-1) - edges(2:end))./(log(edges(1:end-1)) - log(edges(2:end))));
    bar(center, N,'hist');
    % remove the wierd *s
    ax = gca;
    line_plot = ax.Children.findobj('Marker', '*');
    line_plot.delete;
    
    hold on
    set(gca, 'XScale', 'log');
    %p = fitdist(cur_dist', 'BirnbaumSaunders');
    %prob = p.pdf(center);
    %fitcount = prob.*numel(cur_dist).*diff(edges);
    %fitcount = prob;
    averageN=conv(N,ones(1,ws),'same')/ws; %Smooth the bar heights, averaging over 5 bins
    plot(center, averageN, 'LineWidth', 2);
    hold on
    ylabel('Count')
    xlabel('Area (\mu m^2)')
end
