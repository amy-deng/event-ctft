plot(x, y1, type = "b", frame = FALSE, pch = 19, 
     col = "red", xlab = "x", ylab = "y")
# Add a second line
lines(x, y2, pch = 18, col = "blue", type = "b", lty = 2)
# Add a legend to the plot
legend("topleft", legend=c("Line 1", "Line 2"),
       col=c("red", "blue"), lty = 1:2, cex=0.8)



x <- 1:7
# f1 THA
y1 <- c(0.643,0.706 ,0.764,0.821 ,0.839 ,0.858 ,0.885)
y2 <- c(0.625,0.728,0.755,0.796,0.816,0.851,0.875)
ylab <- "F1"
# par(c("mar", "mai"))
dev.new(width=6, height=5, unit="in")
par(mar=c(4.2,4.2,0.3,0.3))
# par(mai=rep(0.4, 4))
# par(mar = c(3, 5, 3, 5))
plot(x, y1, type = "b", frame = FALSE, pch = 19, 
     col = "blue", xlab = "Lead time window (Day)", ylab = "F1",cex.lab=1.1, cex.axis=1.1, cex.main=1.1, cex.sub=1.1)
# Add a second line
lines(x, y2, pch = 15, col = "red", type = "b", lty = 2)
# Add a legend to the plot
legend("topleft", legend=c("Ours", "Ours-causal"),
       col=c("red", "blue"), lty = 1:2, cex=1.1, box.lty=0)



lineplot.function <- function(x,y1,y2,ylab,legpos) {
   # par(c("mar", "mai"))
    dev.new(width=4.8, height=4.2, unit="in")
    par(mar=c(4.2,4.2,1,1))
    # par(mai=rep(0.4, 4))
    # par(mar = c(3, 5, 3, 5))
    plot(x, y1, type = "b", frame = FALSE, pch = 19,  panel.first=grid(),
        col = "blue", xlab = "Lead time window (Day)", ylab = ylab,cex.lab=1.3, cex.axis=1.3, cex.main=1.3, cex.sub=1.3,
        # ylim=c(0.6,0.9)
        )
    # Add a second line
    lines(x, y2, pch = 15, col = "red", type = "b", lty = 2)
    # Add a legend to the plot
    legend(legpos, legend=c("Ours","Ours w/o causal"),
        col=c("blue", "red"), lty = 1:2, cex=1.3, box.lty=0)
}



lineplot2.function <- function(x,y1,y2,ylab,legpos) {
   # par(c("mar", "mai"))
    dev.new(width=4, height=4.8, unit="in")
    par(mar=c(4.2,4.2,0.3,0.3))
    # par(mai=rep(0.4, 4))
    # par(mar = c(3, 5, 3, 5))
    plot(x, y1, type = "b", frame = FALSE, pch = 19, 
        col = "blue", xlab = "Lead time window (Day)", ylab = ylab,cex.lab=1.3, cex.axis=1.3, cex.main=1.3, cex.sub=1.3,
        # ylim=c(0.7,0.9), xpd = FALSE,
        )
    # Add a second line
    lines(x, y2, pch = 15, col = "red", type = "b", lty = 2)
    # Add a legend to the plot
    legend(legpos, legend=c("Ours","Ours w/o causal"),
        col=c("blue", "red"), lty = 1:2, cex=1.3, box.lty=0)
}

lineplot3.function <- function(x,y1,y2,y3,ylab,legpos) {
   # par(c("mar", "mai"))
    dev.new(width=4.8, height=4.2, unit="in")
    par(mar=c(4.2,4.2,1,1))
    # par(mai=rep(0.4, 4))
    # par(mar = c(3, 5, 3, 5))
    plot(x, y1, type = "b", frame = FALSE, pch = 19,  panel.first=grid(),
        col = "blue", xlab = "Lead time window (Day)", ylab = ylab,cex.lab=1.3, cex.axis=1.3, cex.main=1.3, cex.sub=1.3,
        # ylim=c(0.6,0.9)
        )
    # Add a second line
    lines(x, y2, pch = 15, col = "red", type = "b", lty = 2)
    lines(x, y3, pch = 21, col = "orange", type = "b", lty = 2)
    # Add a legend to the plot
    legend(legpos, legend=c("Ours","Ours w/o causal", "HGT"),
        col=c("blue", "red",'orange'), lty = 1:2, cex=1.3, box.lty=0)
}
# http://www.sthda.com/english/wiki/add-legends-to-plots-in-r-software-the-easiest-way
x <- 1:7
ylab <- "F1"
# f1 THA
y1 <- c(0.643,0.706 ,0.764,0.821 ,0.839 ,0.858 ,0.885)
y2 <- c(0.625,0.728,0.755,0.796,0.816,0.851,0.875)
lineplot.function(x,y1,y2,ylab,legpos)

x <- 2:7
y1 <- c(0.706 ,0.764,0.821 ,0.839 ,0.858 ,0.885)
y2 <- c(0.728,0.755,0.796,0.816,0.851,0.875)
lineplot.function(x,y1,y2,ylab,legpos)

legpos <- "bottomright"

y1 <- c(0.706 ,0.764,0.821 ,0.839 ,0.858 ,0.885)
y2 <- c(0.728,0.755,0.796,0.816,0.851,0.875)
y3 <- c(0.673, 0.74, 0.759, 0.803, 0.839, 0.88)
lineplot3.function(x,y1,y2,y3,ylab,legpos)


# AFG
y1 <- c(0.122,0.388 ,0.614 ,0.648 ,0.7   ,0.744 ,0.799)
y2 <- c(0.0 ,0.38  ,0.569 ,0.639 ,0.678,0.742 ,0.795)
lineplot.function(x,y1,y2,ylab,legpos)



y1 <- c(0.388 ,0.614 ,0.648 ,0.7   ,0.744 ,0.799)
y2 <- c(0.38  ,0.569 ,0.639 ,0.678,0.742 ,0.795)
lineplot.function(x,y1,y2,ylab,legpos)

y3 <- c(0.378, 0.552, 0.615, 0.65, 0.72, 0.757)
lineplot3.function(x,y1,y2,y3,ylab,legpos)



# RUS
y1 <- c(0.166,0.646,0.769,0.819,0.86 ,0.884,0.911)
y2 <- c(0.092, 0.63,  0.754, 0.806, 0.854, 0.882, 0.911)
legpos <- "bottomright"
lineplot.function(x,y1,y2,ylab,legpos)
x <- 2:7
y1 <- c(0.646,0.769,0.819,0.86 ,0.884,0.911)
y2 <- c(0.63,  0.754, 0.806, 0.854, 0.882, 0.911)
legpos <- "bottomright"
lineplot.function(x,y1,y2,ylab,legpos)

y3 <- c(0.614, 0.729, 0.787, 0.842, 0.874, 0.902)
lineplot3.function(x,y1,y2,y3,ylab,legpos)

# RUS BACC
y1 <- c(0.54  ,0.769 ,0.827 ,0.859 ,0.883 ,0.893 ,0.914)
y2 <- c( 0.529,0.76,0.816,0.847,0.877,0.892,0.914)
ylab <- "BACC"
lineplot.function(x,y1,y2,ylab)

 
# EGY
y1 <- c(0.589,0.761,0.838,0.878,0.897,0.913,0.927)
y2 <- c(0.585,0.748,0.837,0.864,0.895,0.909,0.924)
legpos <- "bottomright"
lineplot.function(x,y1,y2,ylab,legpos)

x <- 2:7
y1 <- c(0.761,0.838,0.878,0.897,0.913,0.927)
y2 <- c(0.748,0.837,0.864,0.895,0.909,0.924)
lineplot.function(x,y1,y2,ylab,legpos)

y3 <- c(0.745, 0.831, 0.866, 0.882, 0.907, 0.925)
lineplot3.function(x,y1,y2,y3,ylab,legpos)



barplot.function <- function(y,ylab,min,max) { 
    dev.new(width=4.8, height=4.2, unit="in")
    par(mar=c(2.5,4,1.5,1))
    barplot(y,  horiz=FALSE, #main="Car Distribution",
    names.arg=c("Ours-3", "Ours-7", "Ours-14", "Ours"),ylim = c(min, max), xpd = FALSE, ylab = ylab,
    # width=c(0.2,0.2,0.2,0.2), 
    col=c("brown","brown","brown","blue"),
    # density=c(30,20,10,5), 
    # angle=c(11,90,45,0)
    ,space=c(0.7,0.7,0.7,1.4) 
    )  
}

barerrorplot.function <- function(y,errors,ylab,min,max) { 
    dev.new(width=4.8, height=4.2, unit="in")
    par(mar=c(2.5,4,1.5,1))
    barCenters <- barplot(y,  horiz=FALSE, panel.first=grid(),#main="Car Distribution",
    names.arg=c("Ours-3", "Ours-7", "Ours-14", "Ours"),ylim = c(min, max), xpd = FALSE, ylab = ylab,
    # width=c(0.2,0.2,0.2,0.2), 
    col=c("brown","brown","brown","blue"),
    # density=c(30,20,10,5), 
    # angle=c(11,90,45,0)
    ,space=c(0.7,0.7,0.7,1.4) 
    )  
    segments(barCenters, y-errors, barCenters, y+errors, lwd=1)
}

barerrorplotall.function <- function(y,errors,ylab,min,max) { 
    dev.new(width=4.8, height=4.2, unit="in")
    par(mar=c(2.5,4,1.5,1))
    barCenters <- barplot(y,  horiz=FALSE, panel.first=grid(),#main="Car Distribution",
    names.arg=c("N/A","3", "7", "14", "ALL"),ylim = c(min, max), xpd = FALSE, ylab = ylab,
    # width=c(0.2,0.2,0.2,0.2), 
    col=c("yellow","lightblue","lightblue","lightblue","blue"),
    # density=c(30,20,10,5), 
    # angle=c(11,90,45,0)
    ,space=c(1.4,1.4,0.7,0.7,0.7) 
    )  
    segments(barCenters, y-errors, barCenters, y+errors, lwd=0.5)
    abline(h=y[1],lty = '2947', lwd=0.5)

}
# smaller
barerrorplotall_smaller.function <- function(y,errors,ylab,min,max) { 
    dev.new(width=3.6, height=3.0, unit="in")
    par(mar=c(2.5,4,0.5,0.6))
    barCenters <- barplot(y,  horiz=FALSE, panel.first=grid(),#main="Car Distribution",
    names.arg=c("N/A","3", "7", "14", "ALL"),ylim = c(min, max), xpd = FALSE, ylab = ylab, axes=TRUE,
    # width=c(0.2,0.2,0.2,0.2), 
    col=c("#deebf7","#9ecae1","#9ecae1","#9ecae1","#3182bd"),
    # density=c(30,20,10,5), 
    # angle=c(11,90,45,0)
    ,space=c(0.8,0.8,0.8,0.8,0.8) 
    )  
    segments(barCenters, y-errors, barCenters, y+errors, lwd=0.5)
    abline(h=y[1],lty = '2947', lwd=0.5)
    box(bty="l")
    # mtext(2, text = "Trip Frequency", line = 2, las = 1)
}
barerrorplotall_smaller2.function <- function(y,errors,ylab,min,max) { 
    dev.new(width=4, height=3.0, unit="in")
    par(mar=c(2.5,4,0.5,0.6))
    barCenters <- barplot(y,  horiz=FALSE, panel.first=grid(),#main="Car Distribution",
    names.arg=c("N/A","3", "7", "14", "ALL"),ylim = c(min, max), xpd = FALSE, ylab = ylab,
    # width=c(0.2,0.2,0.2,0.2), 
    col=c("#deebf7","#9ecae1","#9ecae1","#9ecae1","#3182bd"),
    # density=c(30,20,10,5), 
    # angle=c(11,90,45,0)
    ,space=c(0.8,0.8,0.8,0.8,0.8) 
    )  
    segments(barCenters, y-errors, barCenters, y+errors, lwd=0.5)
    abline(h=y[1],lty = '2947', lwd=0.5)
    box(bty="l")

}

}
# barerrorplotall_smaller.function <- function(y,errors,ylab,min,max) { 
#     dev.new(width=5, height=3.6, unit="in")
#     par(mar=c(4,2.2,0.5,0.2))
#     barCenters <- barplot(y,  horiz=TRUE, panel.first=grid(),#main="Car Distribution",
#     names.arg=c("N/A","3", "7", "14", "ALL"),xlim = c(min, max), xpd = FALSE, xlab = ylab,
#     # width=c(0.2,0.2,0.2,0.2), 
#     col=c("#deebf7","#9ecae1","#9ecae1","#9ecae1","#3182bd"),
#     # density=c(30,20,10,5), 
#     # angle=c(11,90,45,0)
#     ,space=c(0.6,0.6,0.6,0.6,0.6) 
#     )  
#     segments(barCenters, y-errors, barCenters, y+errors, lwd=0.5)
#     abline(v=y[1],lty = '2947', lwd=0.5)
# }

ylab <- 'F1'
# THA
counts <- c(0.823,0.828,0.836,0.839)
barplot.function(counts,ylab,0.78,0.84)
counts <- c(0.823,0.828,0.836,0.839)
errors <- c(0.007, 0.021, 0.014, 0.023)
barerrorplot.function(counts,errors,ylab,0.67,0.87)

counts <- c(0.816, 0.823,0.828,0.836,0.839)
errors <- c(0.011, 0.007, 0.021, 0.014, 0.023)
barerrorplotall.function(counts,errors,ylab,0.67,0.87)
barerrorplotall_smaller.function(counts,errors,ylab,0.67,0.87)
barerrorplotall_smaller2.function(counts,errors,ylab,0.7,0.87)




# AFG
counts <- c(0.706,0.692,0.713,0.7)
barplot.function(counts,ylab,0.64,0.72)
counts <- c(0.706,0.692,0.713,0.7)
errors <- c(0.023, 0.017, 0.022, 0.013)
barerrorplot.function(counts,errors,ylab,0.54,0.74)

counts <- c(0.678, 0.706,0.692,0.713,0.7)
errors <- c( 0.04,0.023, 0.017, 0.022, 0.013)
barerrorplotall.function(counts,errors,ylab,0.5,0.74)
barerrorplotall_smaller.function(counts,errors,ylab,0.5,0.74)
barerrorplotall_smaller2.function(counts,errors,ylab,0.56,0.74)


# AFG 64
counts <- c(0.674,0.685,0.713,0.7)
barplot.function(counts,ylab,0.62,0.72)
 


# RUS
counts <- c(0.854,0.848 ,0.869 ,0.86)
barplot.function(counts,ylab,0.8,0.88)

counts <- c(0.854,0.857 ,0.869 ,0.86)
errors <- c(0.01, 0.004, 0.013, 0.006)
barerrorplot.function(counts,errors,ylab,0.7,0.9)

counts <- c(0.854, 0.854,0.857 ,0.869 ,0.86)
errors <- c(0.008, 0.01, 0.004, 0.013, 0.006)
barerrorplotall.function(counts,errors,ylab,0.72,0.9)
barerrorplotall_smaller.function(counts,errors,ylab,0.72,0.89)
barerrorplotall_smaller2.function(counts,errors,ylab,0.78,0.89)

# 0.854 0.01 
# 0.848 0.009
# 0.869 0.013
 
# EGY

counts <- c(0.899,0.906,0.893,0.9)
barplot.function(counts,ylab,0.87,0.91)
counts <- c(0.899,0.906,0.893,0.9)
errors <- c(0.009, 0.007, 0.013, 0.006)
barerrorplot.function(counts,errors,ylab,0.74,0.94)

counts <- c(0.895, 0.899,0.906,0.893,0.9)
errors <- c( 0.004, 0.009, 0.007, 0.013, 0.006)
barerrorplotall.function(counts,errors,ylab,0.76,0.94)
barerrorplotall_smaller.function(counts,errors,ylab,0.76,0.93)
barerrorplotall_smaller2.function(counts,errors,ylab,0.8,0.93)



dev.new(width=4.8, height=4.5, unit="in")
par(mar=c(2.5,4,1.5,1))
barplot(counts,  horiz=FALSE, #main="Car Distribution",
  names.arg=c("Ours-3", "Ours-7", "Ours-14", "Ours"),ylim = c(0.8, 0.84), xpd = FALSE, ylab = 'F1',
#    width=c(0.05,0.05,0.05,0.05), 
   col=c("brown","brown","brown","blue"),
   density=c(30,20,10,5) , angle=c(11,90,45,0),space=c(0.5,0.5,0.5,0.5) )  
#    col=c("red","orange","green","blue"))
    # col="brown" )  
    # density=c(20,20,20,20) , angle=c(45,90,120,0),
# abline(h = 10.2, col = "white", lwd = 2, lty = 2)



barerrorplotall.function <- function(y,errors,ylab,min,max) { 
    dev.new(width=4.8, height=4.2, unit="in")
    par(mar=c(2.5,4,1.5,1))
    barCenters <- barplot(y,  horiz=FALSE, panel.first=grid(),#main="Car Distribution",
    names.arg=c("N/A","3", "7", "14", "ALL"),ylim = c(min, max), xpd = FALSE, ylab = ylab,
    # width=c(0.2,0.2,0.2,0.2), 
    col=c("yellow","lightblue","lightblue","lightblue","blue"),
    # density=c(30,20,10,5), 
    # angle=c(11,90,45,0)
    ,space=c(1.4,1.4,0.7,0.7,0.7) 
    )  
    segments(barCenters, y-errors, barCenters, y+errors, lwd=0.5)
    abline(h=y[1],lty = '2947', lwd=0.5)
}




library(plotly)
x <- c(32, 32, 32, 48, 48, 48, 64, 64, 64, 80, 80, 80)
y <- c(1,  2,   3, 1,  2,   3, 1,  2,   3, 1,  2,   3)
# z <- c(0.796, 0.809, 0.808,  0.839, 0.812, 0.798, 0.828, 0.841, 0.827, 0.828, 0.807, 0.802) # THA
# z <- c(0.7, 0.688, 0.698,  0.685, 0.664, 0.68, 0.675, 0.674, 0.672, 0.669, 0.657, 0.611) # AFG
z <- c(0.899,0.888,0.883, 0.897, 0.898, 0.896, 0.888, 0.887, 0.891, 0.896, 0.896, 0.883 ) # EGY
z <- c(0.86,0.837,0.825,0.853,0.842,0.836,0.855,0.843 ,0.83, 0.857, 0.84, 0.83  ) # RUS

 
 
axx <- list(
  title = "Hidden state"
)

axx <- list(
title = "Hidden state",
  ticktext = c("32", "48", "64", "80"),
  tickvals = c(32,48,64,80),
  range = c(32,80)
)


axy <- list(
  title = "Layer",
  ticktext = c("1", "2", "3"),
  tickvals = c(1,2,3),
  range = c(1,3)
)

axz <- list(
  title = "F1"
)


fig <- plot_ly(x = ~x, y = ~y, z = ~z, intensity = ~z, type = 'mesh3d', colors = colorRamp(c("blue", "lightblue", "chartreuse3", "yellow", "red")))
fig <- fig %>% layout(scene = list(xaxis=axx,yaxis=axy,zaxis=axz))
