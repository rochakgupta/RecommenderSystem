function [arr] = nomf_amsd_calc(index) 

Y = xlsread('matrixamsd1.csv');
cmat = xlsread('clustersprev.csv');
vals = csvread('finalamsdnomfval.csv');
users = find(cmat==index);
Y = Y(users,users);
N = length(users);
Rt = xlsread('ratings.csv');
ex = csvread('check.csv');
ex = ex+1;
Rt = Rt(users,ex);
fprintf('\n');
arr = zeros(5,2);
[Y, Z]=sort(Y,2,'descend');

for k=1:5
    c=0;
    for i=1:N
        for j=1:242
            if(Rt(i,j)>0)
                in=find(Rt(:,j)>0);
                cnt=sum(Y(i,in)>0);
                cnt=cnt-1;
                %fprintf('Users for i %d and j %d : %d\n',i,j,cnt);
                cnt=floor(0.2*k*cnt);
                SimS=0;
                Rs=0;
                f=cnt;
                c=c+1;
                for u=1:N
                    if(f==0)
                        break;
                    end;
                    if(Z(i,u)~=i && Rt(Z(i,u),j)>0)
                        f=f-1;
                        Rs = Rs + Rt(u,j)*Y(i,u);
                        SimS = SimS + Y(i,u);
                    end;
                end;
                if(SimS==0)
                    c=c-1;
                    continue;
                end;
                rating = Rs/SimS;
                arr(k,1) = arr(k,1) + (Rt(i,j)-rating)*(Rt(i,j)-rating);
                arr(k,2) = arr(k,2) + abs(Rt(i,j)-rating);
            end;
        end;
    end
    if(c>0)
        arr(k,1)=sqrt(arr(k,1)/c);
        arr(k,2)=arr(k,2)/c;
        vals(k,index)=c;
    end;
end;
dlmwrite('finalamsdnomfval.csv',vals,',');
% The relative error (without the rounding) is quite low:
%fprintf('Relative error, no rounding: %.8f%%\n', norm(X-Xk,'fro')/norm(X,'fro')*100 );
end