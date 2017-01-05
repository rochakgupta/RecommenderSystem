function mf_acos
    a=zeros(5,12);
    arr=zeros(5,2);
    for i=1:6
        arr = mf_acos_calc(i);
        a(:,2*i-1) = arr(:,1); 
        a(:,2*i) = arr(:,2);
    end;
    display(a);
    dlmwrite('finalacos.csv',a,',');
end
        
        