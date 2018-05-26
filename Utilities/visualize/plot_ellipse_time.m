%load ellipses
t = datetime(time./1000,'ConvertFrom','posixTime','TimeZone','America/New_York','Format','dd-MMM-yyyy HH:mm:ss.SSS');
t = t - t(1);
nbins = 20;


working_on = beta_ellipse;
long_mean = [];
short_mean = [];
for i = 1 : numel(working_on)
    longshort = cell2mat(working_on{i}(:,2));
    long_mean(i) = mean(longshort(:,2));
    short_mean(i) = mean(longshort(:,1));
end
hold on
yyaxis left
plot(t, long_mean)
yyaxis right

plot(t, short_mean)