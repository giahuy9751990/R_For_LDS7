sum_even.function = function(start,end){
    sum_even = 0
    for(i in start:end){
        if(i%%2==0){
            sum_even = sum_even + i
        }
    }
    return (sum_even)
}

drink.function = function(price,type="Tea"){
    print(paste("With",price,"you can drink",type))
}

tinhtiendien.function = function(sokw){
    muc1 = 1678
    muc2 = 1734
    muc3 = 2014
    muc4 = 2536
    muc5 = 2834
    muc6 = 2927
    
    bac50 = 50
    bac100 = 100
    
    tiendien = 0
    if (sokw<=50){
        tiendien = sokw*muc1
    } else if (sokw<=100){
        tiendien = bac50*muc1 + (sokw-bac50)*muc2
    } else if (sokw<=200){
        tiendien = bac50*muc1 + bac50*muc2 + (sokw-bac100)*muc3
    } else if (sokw<=300){
        tiendien = bac50*muc1 + bac50*muc2 + bac100*muc3 + (sokw-bac50-bac50-bac100)*muc4
    } else if (sokw<=400){
        tiendien = bac50*muc1 + bac50*muc2 + bac100*muc3 + bac100*muc4 + (sokw-bac50-bac50-bac100-bac100)*muc5
    } else {
        tiendien = bac50*muc1 + bac50*muc2 + bac100*muc3 + bac100*muc4 + bac100*muc5 + (sokw-bac50-bac50-bac100-bac100-bac100)*muc6
    }
    return tiendien
}
