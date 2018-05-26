%load ellipses
t = datetime(time./1000,'ConvertFrom','posixTime','TimeZone','America/New_York','Format','dd-MMM-yyyy HH:mm:ss.SSS');
t = t - t(1);
nbins = 20;

selected_time = {minutes(10)};
ws = 2;
working_on = beta_ellipse;
long_mean = [];
short_mean = [];
for i = 1 : numel(selected_time)
    index =  find(t >selected_time{i}, 1);
    longshort = cell2mat(working_on{index}(:,2));
    for j = 1:2
        cur_dist = double(longshort(:,j))*1.3;
        cur_dist(cur_dist <1) = [];
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
    end
    ylabel('Count')
    xlabel('Length (\mu m)')
end
