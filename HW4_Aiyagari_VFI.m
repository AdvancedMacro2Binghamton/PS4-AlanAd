% PROGRAM NAME: HW4_Aiyagari_VFI
clear, clc
tic

% PARAMETERS
beta = .99; %discount factor 
sigma = 2; % coefficient of risk aversion
alpha=1/3; % production function
delta= .025; % depreciation
rho_e =.5; % AR1 persistence
sigma_e=.2; % White noise variance

num_z = 5;
[z_grid, PI]= TAUCHEN(num_z, rho_e, sigma_e, 3);
z_grid =exp(z_grid');

% Invariant Transition Distribution Using Eigen-vectors of PI-Matrix
[eig_vec, eig_val]= eig(PI');

% Normalize first eigen-vector to have a sum of 1.
PI_Inv= eig_vec(:,1) ./ sum(eig_vec(:,1));

% ASSET VECTOR
a_lo = 0; %lower bound of grid points
a_hi = 80; %upper bound of grid points
num_a = 50;

a = linspace(a_lo, a_hi, num_a); % asset (row) vector

% Aggregate Labor and Capital

% Agg Labor is integral of z*L*pi_inv over z, L=1
aggL= z_grid*PI_Inv;

% Capital Range 
k_min = 20;
k_max = 500;

K_dist=1;

while abs(K_dist) >= .01
    
    % Set INITIAL K_GUESS, rental rate and wage
    k_guess = (k_min + k_max)/2;
    
    r= alpha*((aggL/k_guess)^(1-alpha)) + (1-delta);
    w= (1-alpha)*((k_guess/aggL)^alpha);
    
    cons = bsxfun(@minus, r*a', a);
    cons = bsxfun(@plus, cons, permute(z_grid, [1 3 2])*w);
    ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
    ret(cons<0) = -Inf;
    
    
    v_guess = zeros(num_z, num_a);
    
    % VALUE FUNCTION ITERATION
v_tol = 1;
while v_tol >.0001;
   % CONSTRUCT TOTAL RETURN FUNCTION
   v_mat = ret + beta * ...
       repmat(permute(PI * v_guess, [3 2 1]), [num_a 1 1]);
   
   % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
   [vfn, pol_indx] = max(v_mat, [], 2);
   vfn = permute(vfn, [3 1 2]);
   
   v_tol = abs(max(v_guess(:) - vfn(:)));
   
   v_guess = vfn; %update value functions
end;

% KEEP DECSISION RULE
pol_indx = permute(pol_indx, [3 1 2]);
p_fn = a(pol_indx);

 % SET UP INITITAL DISTRIBUTION
Mu = zeros(num_z,num_a);
Mu(1, 4) = 1; % initial guess: everyone employed, 0 assets
% Mu = ones(2, num_a); alternative initial guess: same mass in all states
% Mu = Mu_guess / sum(Mu_guess(:)); % normalize total mass to 1

% ITERATE OVER DISTRIBUTIONS
% way 1: loop over all non-zeros states
mu_tol = 1;
while mu_tol > 1e-08
    [emp_ind, a_ind] = find(Mu > 0); % find non-zero indices
    
    MuNew = zeros(size(Mu));
    for ii = 1:length(emp_ind)
        apr_ind = pol_indx(emp_ind(ii), a_ind(ii)); 
        MuNew(:, apr_ind) = MuNew(:, apr_ind) + ...
            (PI(emp_ind(ii), :) * Mu(emp_ind(ii), a_ind(ii)) )';
    end

    mu_tol = max(abs(MuNew(:) - Mu(:)));
    
    Mu = MuNew ;
end  
    
    
% CHECK AGGREGATE Savings = AGG Capital

AggS = sum( p_fn(:) .* Mu(:) ); % Aggregate future assets   

K_dist = AggS - k_guess;

if K_dist > 0 ;
    k_min = k_guess ;
end ;
if K_dist < 0;
    k_max = k_guess ;
end ;
    
display (['Current K Guess = ', num2str(k_guess)])
display (['Aggregate desired wealth = ', num2str(AggS)]);
display (['New Kmin is ', num2str(k_min), ', New Kmax is ', num2str(k_max)]);
display (['New K Guess is ', num2str((k_max + k_min)/2)]);    
display (['Tolerance Level ', num2str(K_dist)]);     

end

runtime= toc;

% Graphs of Policy Function and Value Function

figure(1)
 plot(a,vfn(1,:),'blue')
 hold on
 plot(a,vfn(2,:),'red')
 hold on
 plot(a,vfn(3,:),'black')
 hold on
 plot(a,vfn(4,:),'green')
 hold on
 plot(a,vfn(5,:),'yellow')
 legend('Z= 0.5','Z= 0.7','Z= 1','Z= 1.4', 'Z= 1.9','south')
 title(['Value Function'])
 hold off
 
figure(2)
 plot(a, p_fn(1,:),'blue')
 hold on
 plot(a, p_fn(2,:),'red')
 hold on
 plot(a, p_fn(3,:),'black')
 hold on
 plot(a, p_fn(4,:),'green')
 hold on
 plot(a, p_fn(5,:),'yellow')
 legend('Z= 0.5','Z= 0.7','Z= 1','Z= 1.4', 'Z= 1.9','south')
 title(['Policy Function'])
 hold off
 

 