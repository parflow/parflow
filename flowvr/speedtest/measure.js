#!/usr/bin/node

var fs = require('fs');
var argv = process.argv;
if (process.argv.length > 2) {
    var values = [
        showIt(['./mpi.out.00001.nc', './mpi.out.00999.nc']),
        showIt(['./results_normal/normal.out.00001.nc', './results_normal/normal.out.00999.nc']),
        showIt(['./mpi.out.00001.nc', './mpi.out.00500.nc']),
        showIt(['./results_normal/normal.out.00001.nc', './results_normal/normal.out.00500.nc']),
        fs.statSync('./mpi.out.00001.nc').blocks,
        fs.statSync('./mpi.out.00001.nc').size
    ]
    var out = process.argv.slice(2).concat(values).join(",");
    console.log(out);
} else {
    var files = ['./mpi.out.00001.nc', './mpi.out.00999.nc'];
    console.log("FlowVR runtime all files: " + showIt(files) + " s");
    var files = ['./results_normal/normal.out.00001.nc', './results_normal/normal.out.00999.nc'];
    console.log("normal runtime all files: " + showIt(files) + " s");

    var files = ['./mpi.out.00001.nc', './mpi.out.00500.nc'];
    console.log("FlowVR runtime half files: " + showIt(files) + " s");
    var files = ['./results_normal/normal.out.00001.nc', './results_normal/normal.out.00500.nc'];
    console.log("normal runtime half files: " + showIt(files) + " s");
}

function showIt(files) {
    var times = files.map((file)=>{
        return fs.statSync(file).mtime;
    });
    return Math.abs(times[1] - times[0])/1000
}
