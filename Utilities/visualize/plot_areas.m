%load areas

t = datetime(time./1000,'ConvertFrom','posixTime','TimeZone','America/New_York','Format','dd-MMM-yyyy HH:mm:ss.SSS');
t = t - t(1);
[sorted_t, index] = sort(t);
alpha_dist = alpha_dist(index);
beta_dist = beta_dist(index);
% plot total area change
figure
hold on
yyaxis left
ylabel('\alpha-form crystal total area (\mu m^2)');
acc_alpha = cellfun(@sum,alpha_dist)*1.3*1.3;
%plot(t,smooth(double(acc_alpha)));
plot(sorted_t,acc_alpha./double(counter_list));

yyaxis right
ylabel('\beta-form crystal count total area (\mu m^2)');
acc_beta = cellfun(@sum,beta_dist)*1.3*1.3;
%plot(t,smooth(double(acc_beta)));
plot(sorted_t, acc_beta./double(counter_list));

xlabel('Time (hours)');
%legend({'\alpha-form', '\beta-form'});
box on;