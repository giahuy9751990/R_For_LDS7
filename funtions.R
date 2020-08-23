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