function u = func_training(xx) 

%%%% weights relted to microsphere concept %%%%
www = [0.0265214244093, 0.0265214244093, 0.0265214244093,... 
   0.0199301476312, 0.0199301476312, 0.0199301476312, 0.0199301476312...
    0.0199301476312, 0.0199301476312, 0.0250712367487,...
   0.0250712367487, 0.0250712367487, 0.0250712367487, 0.0250712367487...
    0.0250712367487, 0.0250712367487, 0.0250712367487,...
   0.0250712367487, 0.0250712367487, 0.0250712367487, 0.0250712367487];
%%%%% directions related to microsphere concept %%%%
DDD = zeros(21,3);
DDD(1,:)=[1, 0, 0];
DDD(2,:)=[0, 1, 0];
DDD(3,:)=[0, 0, 1];
DDD(4,:)=[0.707107, 0.707107, 0];
DDD(5,:)=[0.707107, -0.707107, 0];
DDD(6,:)=[0.707107, 0, 0.707107];
DDD(7,:)=[0.707107, 0, -0.707107];
DDD(8,:)=[0, 0.707107, 0.707107];
DDD(9,:)=[0, 0.707107, -0.707107];
DDD(10,:)=[0.387907, 0.387907, 0.830097];
DDD(11,:)=[0.387907, 0.387907, -0.830097];
DDD(12,:)=[0.387907, -0.387907, 0.830097];
DDD(13,:)=[0.387907, 0.387907, -0.830097];
DDD(14,:)=[0.387907, 0.830097, 0.387907];
DDD(15,:)=[0.387907, 0.830097, -0.387907];
DDD(16,:)=[0.387907, -0.830097, 0.387907];
DDD(17,:)=[0.387907, -0.830097, -0.387907];
DDD(18,:)=[0.830097, 0.387907, 0.387907];
DDD(19,:)=[0.830097, 0.387907, -0.387907];
DDD(20,:)=[0.830097, -0.387907, 0.387907];
DDD(21,:)=[0.830097, -0.387907, -0.387907];

%%% Biaxial Data for Training %%%%
stretch_L = [1,1.00570000000000,1.02920000000000,1.05850000000000,1.13510000000000,1.28840000000000,1.47130000000000,1.63060000000000,1.77800000000000,1.82620000000000,1.91360000000000,2.01970000000000,2.14360000000000,2.23190000000000,2.24220000000000,2.33210000000000,2.41460000000000,2.50300000000000,2.57960000000000,2.65620000000000];
stretch_U = [2.65620000000000,2.64440000000000,2.63290000000000,2.61540000000000,2.59180000000000,2.57440000000000,2.53910000000000,2.49800000000000,2.44500000000000,2.38010000000000,2.29180000000000,2.18570000000000,2.02650000000000,1.85540000000000,1.67840000000000,1.48970000000000,1.34220000000000,1.24200000000000];

Stress_L = [0,0.419600000000000,0.821100000000000,1.22260000000000,1.60610000000000,1.91740000000000,2.21060000000000,2.54010000000000,2.86950000000000,2.90260000000000,3.23530000000000,3.60090000000000,3.94840000000000,4.31390000000000,4.35000000000000,4.71590000000000,5.08130000000000,5.46500000000000,5.84870000000000,6.19580000000000];
Stress_U = [6.19580000000000,5.81270000000000,5.41130000000000,5.02810000000000,4.62660000000000,4.22520000000000,3.82350000000000,3.44020000000000,3.05680000000000,2.67330000000000,2.28960000000000,1.92400000000000,1.59450000000000,1.28310000000000,1.00820000000000,0.733200000000000,0.403800000000000,0];



nos_L = size(stretch_L,2);% number of segments for stretch
nos_U = size(stretch_U,2); % number of segments for stretch
y1 = zeros(1,nos_L);% number of segments for stress
y2 = zeros(1,nos_U);% number of segments for stress
lamda_max = zeros(1,21);
beta_max = zeros(1,21);


%%%%% weights related to ANN %%%%
w1 = xx(1);
w2 = xx(2);
w3 = xx(3);
w4 = xx(4);
w5 = xx(5);
w6 = xx(6);
w7 = xx(7);
w8 = xx(8);
w9 = xx(9);
w10 = xx(10);
w11 = xx(11);
w12 = xx(12);
w13 = xx(13);
w14 = xx(14);
w15 = xx(15);
w16 = xx(16);
w17 = xx(17);
w18 = xx(18);
w19 = xx(19);
w20 = xx(20);
w21 = xx(21);
w22 = xx(22);
w23 = xx(23);
w24 = xx(24);
w25 = xx(25);
w26 = xx(26);
w27 = xx(27);
w28 = xx(28);
w29 = xx(29);
w30 = xx(30);
w31 = xx(31);
w32 = xx(32);



for i=1:nos_L
    
    yy = zeros(3,3);
    F = [stretch_L(i) 0 0;0 (stretch_L(i))^1 0;0 0 (stretch_L(i))^-2];

    for j=1:21
        d = DDD(j,:)';
        lamda = sqrt(d'*(F'*F)*d);
        beta = sqrt(d'*(inv(F)*inv(F)')*d);
        if lamda > lamda_max(j)
           
            lamda_max(j) = lamda;
            
        end
        if beta > beta_max(j)
           
            beta_max(j) = beta;
            
        end
        
        NN1 = (www(j)/lamda)*(w4*(w1+w2*lamda_max(j)+w3*lamda)+w8*sin(w5+w6*lamda_max(j)+w7*lamda)+w12*tan(w9+w10*lamda_max(j)+w11*lamda)+w16*exp(w13+w14*lamda_max(j)+w15*lamda))*F*kron(d',d);
        NN2 = -(www(j)/beta)*(w20*exp(w17+w18*beta_max(j)+w19*beta)^1+w24*(w21+w22*beta_max(j)+w23*beta)^3+w28*(w25+w26*beta_max(j)+w27*beta)^5+w32*tan(w29+w30*beta_max(j)+w31*beta))*inv(F)*inv(F)*inv(F)*kron(d',d);
        yy = yy+NN1+NN2;
    end
    
    F_inv =inv(F);
    p = yy(3,3)/F_inv(3,3);
    yy = yy - p*F_inv;
    y1(i)= yy(1,1);
end

for i=1:nos_U
    
    yy = zeros(3,3);
    F = [stretch_U(i) 0 0;0 (stretch_U(i))^1 0;0 0 (stretch_U(i))^-2];

    for j=1:21
        d = DDD(j,:)';
        lamda = sqrt(d'*(F'*F)*d);
        beta = sqrt(d'*(inv(F)*inv(F)')*d);
        if lamda > lamda_max(j)
           
            lamda_max(j) = lamda;
            
        end
        if beta > beta_max(j)
           
            beta_max(j) = beta;
            
        end
        
        NN1 = (www(j)/lamda)*(w4*(w1+w2*lamda_max(j)+w3*lamda)+w8*sin(w5+w6*lamda_max(j)+w7*lamda)+w12*tan(w9+w10*lamda_max(j)+w11*lamda)+w16*exp(w13+w14*lamda_max(j)+w15*lamda))*F*kron(d',d);
        NN2 = -(www(j)/beta)*(w20*exp(w17+w18*beta_max(j)+w19*beta)^1+w24*(w21+w22*beta_max(j)+w23*beta)^3+w28*(w25+w26*beta_max(j)+w27*beta)^5+w32*tan(w29+w30*beta_max(j)+w31*beta))*inv(F)*inv(F)*inv(F)*kron(d',d);
        yy = yy+NN1+NN2;
    end
    
    F_inv =inv(F);
    p = yy(3,3)/F_inv(3,3);
    yy = yy - p*F_inv;
    y2(i)= yy(1,1);
end

error_L = sum((y1-Stress_L).^2);
error_U = sum((y2-Stress_U).^2);

u = error_L + error_U;
end
