(* synthesize *)
module mkHelloWorldOnce ();
    Reg#(Bool) said <- mkReg(False);
    
    rule sayhello (!said);
        $display("hello, world");
        said <= True;
    endrule

    rule goodbye (said);
        $finish();
    endrule
endmodule
