clear; clc; close all;

%Raw data import
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('trial_step.mat')

UU = [];
YY = [];
k = 1;
tmp = [1,2,3,4,5,6];

for q = 1:length(tmp)
    %왼발
    if tmp(q) ~= 2 && tmp(q) ~= 4
        for i = 1:11
            for j = 1:length(trial_step{i, tmp(q)}.emg.left(:,1))
                U(:,k) = trial_step{i, tmp(q)}.emg.left(j,:)';
                Y(:,k) = trial_step{i, tmp(q)}.imu.left.middle.gyro.x(j,:)';
                k = k + 1;
            end
        end
        %오른발
        for i = 1:11
            for j = 1:length(trial_step{i, tmp(q)}.emg.right(:,1))
                U(:,k) = trial_step{i, tmp(q)}.emg.right(j,:)';
                Y(:,k) = trial_step{i, tmp(q)}.imu.right.middle.gyro.x(j,:)';
                k = k + 1;
            end
        end
        
    elseif tmp(q) == 2 %Subject 2는 i가 3이하에서 잘못된 데이터
        %왼발
        for i = 4:11
            for j = 1:length(trial_step{i, tmp(q)}.emg.left(:,1))
                U(:,k) = trial_step{i, tmp(q)}.emg.left(j,:)';
                Y(:,k) = trial_step{i, tmp(q)}.imu.left.middle.gyro.x(j,:)';
                k = k + 1;
            end
        end
        %오른발
        for i = 4:11
            for j = 1:length(trial_step{i, tmp(q)}.emg.right(:,1))
                U(:,k) = trial_step{i, tmp(q)}.emg.right(j,:)';
                Y(:,k) = trial_step{i, tmp(q)}.imu.right.middle.gyro.x(j,:)';
                k = k + 1;
            end
        end
        
    elseif tmp(q) == 4 %Subject 4는 i가 2이하에서 잘못된 데이터
        for i = 3:11
            for j = 1:length(trial_step{i, tmp(q)}.emg.left(:,1))
                U(:,k) = trial_step{i, tmp(q)}.emg.left(j,:)';
                Y(:,k) = trial_step{i, tmp(q)}.imu.left.middle.gyro.x(j,:)';
                k = k + 1;
            end
        end
        %오른발
        for i = 3:11
            for j = 1:length(trial_step{i, tmp(q)}.emg.right(:,1))
                U(:,k) = trial_step{i, tmp(q)}.emg.right(j,:)';
                Y(:,k) = trial_step{i, tmp(q)}.imu.right.middle.gyro.x(j,:)';
                k = k + 1;
            end
        end
        
    end
    
    
    
    MU(q,1) = max(max(U)); %EMG data 최대값
    MY(q,1) = max(max(Y)); %Angular velocity data 최대값
    data_number(q,1) = length(U(1,:));
    
    U = U ./ MU(q,1);
    Y = Y ./ MY(q,1);
    
    UU = [UU,U];
    YY = [YY,Y];
    
    clear U;
    clear Y;
    k = 1;
end

%초기값 설정
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = length(UU(1,:)); %입력 데이터 수

NUM = 8;  %epoch 수

alpha = 0.01;

index =  randperm (N);
Train_index = index(1, 1:floor(N*0.8));
Test_index = index(1, length(Train_index)+1:end);

%%%% weight 초기화
W1 = normrnd(0,sqrt(6/1000),[1000, 3000])*0.01;
W2 = normrnd(0,sqrt(6/2000),[3000, 3000])*0.01;
W3 = normrnd(0,sqrt(6/2000),[3000, 3000])*0.01;
W4 = normrnd(0,sqrt(6/2000),[3000, 1000])*0.01;

a = 0.001;

b1  = ones(3000,1)*a;
b2  = ones(3000,1)*a;
b3  = ones(3000,1)*a;
b4  = ones(1000,1)*a;


%Trian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for epoch = 1:NUM
    
    for k = 1:length(Train_index)
        
        u = UU(:,Train_index(k));
        y = YY(:,Train_index(k));
        
        z1 = W1'*u + b1;
        %x1 = Sigmoid(z1);
        x1 = ReLU(z1);
        %         x1 = x1.* Dropout(x1,0.2);
        
        z2 = W2'*x1+ b2;
        x2 = ReLU(z2);
        x2 = x2.* Dropout(x2,0.2);
        
        z3 = W3'*x2+ b3;
        x3 = ReLU(z3);
        x3 = x3.* Dropout(x3,0.2);
        
        z4 = W4'*x3+ b4;
        yhat = tanh(z4);
        
        
        e = yhat - y;
        
        delta = e;
        
        e3 = W4*delta;
        delta3 = (x3>0).*e3;
        e2 = W3*delta3;
        delta2 = (x2>0).*e2;
        e1 = W2*delta2;
        %delta1 = x1.*(1-x1).*e1;
        delta1 = (x1>0).*e1;
        
        dW4 = alpha*x3*delta';
        W4 = W4 - dW4;
        dW3 = alpha*x2*delta3';
        W3 = W3 - dW3;
        dW2 = alpha*x1*delta2';
        W2 = W2 - dW2;
        dW1 = alpha*u*delta1';
        W1 = W1 - dW1;
        
        
        b4 = b4 - alpha * delta;
        b3 = b3 - alpha * delta3;
        b2 = b2 - alpha * delta2;
        b1 = b1 - alpha * delta1;
        
        tmp = e.^2;
        err = sum(tmp);
        
    end
    
    err_log(epoch,1) = err;
    err = 0;
    
    disp(epoch);
end

%Test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k = 1:length(Test_index)
    u = UU(:,Test_index(k));
    y = YY(:,Test_index(k));
    
    z1 = W1'*u + b1;
    x1 = ReLU(z1);
    z2 = W2'*x1+ b2;
    x2 = ReLU(z2);
    z3 = W3'*x2+ b3;
    x3 = ReLU(z3);
    z4 = W4'*x3+ b4;
    yhat = tanh(z4);
    
    e = yhat - y;
    
    tmp = e.^2;
    err = sum(tmp);
    
    error(k,1) = err;
    
    err = 0;
    
    Result(:,k) = yhat;
end

%성능평가
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%랜덤으로 3개 결과를 Actual과 비교
Test_index2 = Test_index(1,1:3);
figure;
for i = 1:3
    
    %Normalize 환원
    if Test_index2(i) <= data_number(1,1)
        Result2(:,i) =  Result(:,i).* MY(1);
        Result3(:,i) =  YY(:,Test_index2(i)).* MY(1);
    elseif Test_index2(i) <= data_number(2,1) &&  Test_index2(i) > data_number(1,1)
        Result2(:,i) =  Result(:,i).* MY(2);
        Result3(:,i) =  YY(:,Test_index2(i)).* MY(2);
    elseif   Test_index2(i) <= data_number(3,1) &&  Test_index2(i) > data_number(2,1)
        Result2(:,i) =  Result(:,i).* MY(3);
        Result3(:,i) =  YY(:,Test_index2(i)).* MY(3);
    elseif   Test_index2(i) <= data_number(4,1) &&  Test_index2(i) > data_number(3,1)
        Result2(:,i) =  Result(:,i).* MY(4);
        Result3(:,i) =  YY(:,Test_index2(i)).* MY(4);
    elseif   Test_index2(i) <= data_number(5,1) &&  Test_index2(i) > data_number(4,1)
        Result2(:,i) =  Result(:,i).* MY(5);
        Result3(:,i) =  YY(:,Test_index2(i)).* MY(5);
    elseif   Test_index2(i) > data_number(5,1)
        Result2(:,i) =  Result(:,i).* MY(6);
        Result3(:,i) =  YY(:,Test_index2(i)).* MY(6);
        
    end
    subplot(3,2,2*i-1);
    plot(Result2(:,i),'r','Linewidth',3);
    ylim([-250 500]);
    if i == 1
                title('Result');
    end
    subplot(3,2,2*i);
    plot(Result3(:,i),'b','Linewidth',3);
    ylim([-250 500]);
    if i == 1
                title('Actual');
    end
end

app = (error < 10);
accuracy = sum(app)/length(error) *100;

accuracy
mean(error)
max(error)


