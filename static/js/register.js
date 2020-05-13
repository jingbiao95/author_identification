$(function () {

    $('#submitButton').click(function () {
        submit();
    });

    function submit() {
        var data = {
            "method": "decisiontree",
            "dataSet": "1101", //提供编号
            "parameters": {
                "depth": "5"
            },
        };
        $.ajax({
            type: 'POST',
            url: '/classification/submit',
            data: JSON.stringify(data), //json对象转化为字符串
            //request Header注明content-type:'application/json; charset=UTF-8'
            contentType: 'application/json; charset=UTF-8',
            dataType: 'json', // 注意：期望服务端返回json格式的数据
            success: function(data) { // 这里的data就是json格式的数据
                alert("success");

            },
            error: function(xhr, type) {
                alert("ss");
            }
        });
    }


})