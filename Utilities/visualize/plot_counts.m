%load counts

t = datetime(time./1000,'ConvertFrom','posixTime','TimeZone','America/New_York','Format','dd-MMM-yyyy HH:mm:ss.SSS');
t = t - t(1);
[sorted_t, index] = sort(t);
sorted_alpha = alpha_count(index);
sorted_beta = beta_count(index);
figure
hold on
yyaxis left
ylabel('\alpha-form crystal count');
%plot(t,smooth(double(alpha_count)));
plot(sorted_t,sorted_alpha);

yyaxis right
ylabel('\beta-form crystal count');
%plot(t,smooth(double(beta_count)));
plot(sorted_t,sorted_beta);

xlabel('Time (hours)');
%legend({'\alpha-form', '\beta-form'});
box on;