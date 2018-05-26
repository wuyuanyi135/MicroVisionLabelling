load counts

t = datetime(time./1000,'ConvertFrom','posixTime','TimeZone','America/New_York','Format','dd-MMM-yyyy HH:mm:ss.SSS');
t = t - t(1);

figure
hold on
yyaxis left
ylabel('\alpha-form crystal count');
%plot(t,smooth(double(alpha_count)));
plot(t,alpha_count);

yyaxis right
ylabel('\beta-form crystal count');
%plot(t,smooth(double(beta_count)));
plot(t,beta_count);

xlabel('Time (hours)');
%legend({'\alpha-form', '\beta-form'});
box on;