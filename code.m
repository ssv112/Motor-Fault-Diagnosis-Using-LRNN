clc; clear all
tic
load('TrainData.mat')
y=training';
t=trainingS1';

%yy=testing';
pp = con2seq((testing)');
tt = con2seq((testingS1)');

p = con2seq(y);
t = con2seq(t);
lrn_net = layrecnet(1,3);
lrn_net.trainFcn = 'traincgb';
lrn_net.trainParam.show = 5;
lrn_net.trainParam.epochs = 100;
lrn_net = train(lrn_net,p,t);
%after training the model
y = lrn_net(p);
%Train output plot
figure(1)
plot(cell2mat(y))
hold on
plot(cell2mat(t))
hold off

%% Test the Model
Y_test = sim(lrn_net,pp);
YY_test=cell2mat(Y_test);
YY_testT=cell2mat(tt);
% plot(cell2mat(Y_test))

% Testing output plot
figure(2)
plot(cell2mat(Y_test))
hold on
plot(cell2mat(tt))
hold off

figure(3)
plotregression(YY_testT,YY_test,'Regresion Test')

YY_test=cell2mat(Y_test);
YY_testT=cell2mat(tt);
error_test=YY_testT-YY_test;
%MSE_test=mse(error_test)
%plotregression()
Toc
