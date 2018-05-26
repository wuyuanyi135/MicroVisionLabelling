load areoldas

t = datetime(time./1000,'ConvertFrom','posixTime','TimeZone','America/New_York','Format','dd-MMM-yyyy HH:mm:ss.SSS');
t = t - t(1);

% plot total area change
figure
hold on
yyaxis left
ylabel('\alpha-form crystal total area (\mu m^2)');
acc_alpha = cellfun(@sum,alpha_dist)*1.3*1.3;
%plot(t,smooth(double(acc_alpha)));
plot(t,acc_alpha);

yyaxis right
ylabel('\beta-form crystal count total area (\mu m^2)');
acc_beta = cellfun(@sum,beta_dist)*1.3*1.3;
%plot(t,smooth(double(acc_beta)));
plot(t, acc_beta);

xlabel('Time (hours)');
%legend({'\alpha-form', '\beta-form'});
box on;