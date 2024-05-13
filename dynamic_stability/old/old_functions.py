    def solve_derivatives2(self, du, dv, dw, dp, dq, dr):
        
        u_o = self.eq_velo[0]
        v_o = self.eq_velo[1]
        w_o = self.eq_velo[2]
        
        V_o = np.sqrt(u_o*u_o + v_o*v_o + w_o*w_o)
        
        p_o = (self.eq_rot[0])
        q_o = (self.eq_rot[1])
        r_o = (self.eq_rot[2])
                
        # phi_o = self.eq_euler[0]
        # theta_o = self.eq_euler[1]
        
        tau_o = self.eq_inputs[0]
        alpha_o = (self.eq_inputs[1])
        beta_o = (self.eq_inputs[2])
        da_o = (self.eq_inputs[3])
        de_o = (self.eq_inputs[4])
        dr_o = (self.eq_inputs[5])
        
        pbar_o = p_o*self.bw/(2.*V_o)
        qbar_o = q_o*self.cw/(2.*V_o)
        rbar_o = r_o*self.bw/(2.*V_o)
        
        u_array = np.array([u_o - 2*du, u_o - du, u_o + du, u_o + 2*du])
        
        FM_du = np.zeros((4,6))
        
        for i in range(len(u_array)):
            V = np.sqrt(u_array[i]*u_array[i] + v_o*v_o + w_o*w_o)
            
            pbar = p_o*self.bw/(2.*V)
            qbar = q_o*self.cw/(2.*V)
            rbar = r_o*self.bw/(2.*V)
            
            alpha = np.arctan2(w_o, u_array[i])
            beta = np.arcsin(v_o/V)
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha, beta, pbar, qbar, rbar, da_o, de_o, dr_o]), tau_o, V) 
            
            FM_du[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])

        self.Fxb_u = self.force_derivs(FM_du[0,0],FM_du[1,0],FM_du[2,0],FM_du[3,0], du)
        self.Fyb_u = self.force_derivs(FM_du[0,1],FM_du[1,1],FM_du[2,1],FM_du[3,1], du)
        self.Fzb_u = self.force_derivs(FM_du[0,2],FM_du[1,2],FM_du[2,2],FM_du[3,2], du)
            
        self.Mxb_u = self.force_derivs(FM_du[0,3],FM_du[1,3],FM_du[2,3],FM_du[3,3], du)
        self.Myb_u = self.force_derivs(FM_du[0,4],FM_du[1,4],FM_du[2,4],FM_du[3,4], du)
        self.Mzb_u = self.force_derivs(FM_du[0,5],FM_du[1,5],FM_du[2,5],FM_du[3,5], du)
        
        print('dFX_du: ', self.Fxb_u)
        print('dFY_du: ', self.Fyb_u)
        print('dFZ_du: ', self.Fzb_u)
        print('dMX_du: ', self.Mxb_u)
        print('dMY_du: ', self.Myb_u)
        print('dMZ_du: ', self.Mzb_u)
        
        v_array = np.array([v_o - 2*dv, v_o - dv, v_o + dv, v_o + 2*dv])
        
        FM_dv = np.zeros((4,6))
        
        for i in range(len(u_array)):
            V = np.sqrt(u_o*u_o + v_array[i]*v_array[i] + w_o*w_o)
            
            pbar = p_o*self.bw/(2.*V)
            qbar = q_o*self.cw/(2.*V)
            rbar = r_o*self.bw/(2.*V)
            
            alpha = np.arctan2(w_o, u_o)
            beta = np.arcsin(v_array[i]/V)
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha, beta, pbar, qbar, rbar, da_o, de_o, dr_o]), tau_o, V) 
            
            FM_dv[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])


        self.Fxb_v = self.force_derivs(FM_dv[0,0],FM_dv[1,0],FM_dv[2,0],FM_dv[3,0], dv)
        self.Fyb_v = self.force_derivs(FM_dv[0,1],FM_dv[1,1],FM_dv[2,1],FM_dv[3,1], dv)
        self.Fzb_v = self.force_derivs(FM_dv[0,2],FM_dv[1,2],FM_dv[2,2],FM_dv[3,2], dv)
            
        self.Mxb_v = self.force_derivs(FM_dv[0,3],FM_dv[1,3],FM_dv[2,3],FM_dv[3,3], dv)
        self.Myb_v = self.force_derivs(FM_dv[0,4],FM_dv[1,4],FM_dv[2,4],FM_dv[3,4], dv)
        self.Mzb_v = self.force_derivs(FM_dv[0,5],FM_dv[1,5],FM_dv[2,5],FM_dv[3,5], dv)
        
        
        w_array = np.array([w_o - 2*dw, w_o - dw, w_o + dw, w_o + 2*dw])
        
        FM_dw = np.zeros((4,6))
        
        for i in range(len(u_array)):
            V = np.sqrt(u_o*u_o + w_array[i]*w_array[i] + v_o*v_o)
            
            pbar = p_o*self.bw/(2.*V)
            qbar = q_o*self.cw/(2.*V)
            rbar = r_o*self.bw/(2.*V)
            
            alpha = np.arctan2(w_array[i], u_o)
            beta = np.arcsin(v_o/V)
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha, beta, pbar, qbar, rbar, da_o, de_o, dr_o]), tau_o, V) 
            
            FM_dw[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])


        self.Fxb_w = self.force_derivs(FM_dw[0,0],FM_dw[1,0],FM_dw[2,0],FM_dw[3,0], dw)
        self.Fyb_w = self.force_derivs(FM_dw[0,1],FM_dw[1,1],FM_dw[2,1],FM_dw[3,1], dw)
        self.Fzb_w = self.force_derivs(FM_dw[0,2],FM_dw[1,2],FM_dw[2,2],FM_dw[3,2], dw)
            
        self.Mxb_w = self.force_derivs(FM_dw[0,3],FM_dw[1,3],FM_dw[2,3],FM_dw[3,3], dw)
        self.Myb_w = self.force_derivs(FM_dw[0,4],FM_dw[1,4],FM_dw[2,4],FM_dw[3,4], dw)
        self.Mzb_w = self.force_derivs(FM_dw[0,5],FM_dw[1,5],FM_dw[2,5],FM_dw[3,5], dw)
        
        
        '''delta rotation rate Data'''
                
        dpbar = dp*self.bw/(2.*V_o)
        dqbar = dq*self.cw/(2.*V_o)
        drbar = dr*self.bw/(2.*V_o)
        
        
        dPbar_array = np.array([pbar_o - 2*dpbar, pbar_o - dpbar, pbar_o + dpbar, pbar_o + 2*dpbar])
        dQbar_array = np.array([qbar_o - 2*dqbar, qbar_o - dqbar, qbar_o + dqbar, qbar_o + 2*dqbar])
        dRbar_array = np.array([rbar_o - 2*drbar, rbar_o - drbar, rbar_o + drbar, rbar_o + 2*drbar])
        
        FM_dPbar = np.zeros((4,6))
        FM_dQbar = np.zeros((4,6))
        FM_dRbar = np.zeros((4,6))
        
        for i in range(len(dPbar_array)):
            
            FXp, FYp, FZp, Mxp, Myp, Mzp = self.run_FM_body(np.array([alpha_o, beta_o, dPbar_array[i], qbar_o, rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            FM_dPbar[i,:] = np.array([FXp, FYp, FZp, Mxp, Myp, Mzp])
            
            FXq, FYq, FZq, Mxq, Myq, Mzq = self.run_FM_body(np.array([alpha_o, beta_o, pbar_o, dQbar_array[i], rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            FM_dQbar[i,:] = np.array([FXq, FYq, FZq, Mxq, Myq, Mzq])
            
            FXr, FYr, FZr, Mxr, Myr, Mzr = self.run_FM_body(np.array([alpha_o, beta_o, pbar_o, qbar_o, dRbar_array[i], da_o, de_o, dr_o]), tau_o, self.V) 
            FM_dRbar[i,:] = np.array([FXr, FYr, FZr, Mxr, Myr, Mzr])
        
        self.Fxb_p = self.force_derivs(FM_dPbar[0,0],FM_dPbar[1,0],FM_dPbar[2,0],FM_dPbar[3,0], dp)
        self.Fyb_p = self.force_derivs(FM_dPbar[0,1],FM_dPbar[1,1],FM_dPbar[2,1],FM_dPbar[3,1], dp)
        self.Fzb_p = self.force_derivs(FM_dPbar[0,2],FM_dPbar[1,2],FM_dPbar[2,2],FM_dPbar[3,2], dp)
            
        self.Mxb_p = self.force_derivs(FM_dPbar[0,3],FM_dPbar[1,3],FM_dPbar[2,3],FM_dPbar[3,3], dp)
        self.Myb_p = self.force_derivs(FM_dPbar[0,4],FM_dPbar[1,4],FM_dPbar[2,4],FM_dPbar[3,4], dp)
        self.Mzb_p = self.force_derivs(FM_dPbar[0,5],FM_dPbar[1,5],FM_dPbar[2,5],FM_dPbar[3,5], dp)
        
        self.Fxb_q = self.force_derivs(FM_dQbar[0,0],FM_dQbar[1,0],FM_dQbar[2,0],FM_dQbar[3,0], dq)
        self.Fyb_q = self.force_derivs(FM_dQbar[0,1],FM_dQbar[1,1],FM_dQbar[2,1],FM_dQbar[3,1], dq)
        self.Fzb_q = self.force_derivs(FM_dQbar[0,2],FM_dQbar[1,2],FM_dQbar[2,2],FM_dQbar[3,2], dq)
            
        self.Mxb_q = self.force_derivs(FM_dQbar[0,3],FM_dQbar[1,3],FM_dQbar[2,3],FM_dQbar[3,3], dq)
        self.Myb_q = self.force_derivs(FM_dQbar[0,4],FM_dQbar[1,4],FM_dQbar[2,4],FM_dQbar[3,4], dq)
        self.Mzb_q = self.force_derivs(FM_dQbar[0,5],FM_dQbar[1,5],FM_dQbar[2,5],FM_dQbar[3,5], dq)
        
        self.Fxb_r = self.force_derivs(FM_dRbar[0,0],FM_dRbar[1,0],FM_dRbar[2,0],FM_dRbar[3,0], dr)
        self.Fyb_r = self.force_derivs(FM_dRbar[0,1],FM_dRbar[1,1],FM_dRbar[2,1],FM_dRbar[3,1], dr)
        self.Fzb_r = self.force_derivs(FM_dRbar[0,2],FM_dRbar[1,2],FM_dRbar[2,2],FM_dRbar[3,2], dr)
            
        self.Mxb_r = self.force_derivs(FM_dRbar[0,3],FM_dRbar[1,3],FM_dRbar[2,3],FM_dRbar[3,3], dr)
        self.Myb_r = self.force_derivs(FM_dRbar[0,4],FM_dRbar[1,4],FM_dRbar[2,4],FM_dRbar[3,4], dr)
        self.Mzb_r = self.force_derivs(FM_dRbar[0,5],FM_dRbar[1,5],FM_dRbar[2,5],FM_dRbar[3,5], dr)
        
        
        print('{:<16}{:<20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,u derivatvies:',self.Fxb_u,self.Fyb_u,self.Fzb_u,self.Mxb_u,self.Myb_u,self.Mzb_u))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,v derivatvies:',self.Fxb_v,self.Fyb_v,self.Fzb_v,self.Mxb_v,self.Myb_v,self.Mzb_v))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,w derivatvies:',self.Fxb_w,self.Fyb_w,self.Fzb_w,self.Mxb_w,self.Myb_w,self.Mzb_w))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,p derivatvies:',self.Fxb_p,self.Fyb_p,self.Fzb_p,self.Mxb_p,self.Myb_p,self.Mzb_p))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,q derivatvies:',self.Fxb_q,self.Fyb_q,self.Fzb_q,self.Mxb_q,self.Myb_q,self.Mzb_q))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,r derivatvies:',self.Fxb_r,self.Fyb_r,self.Fzb_r,self.Mxb_r,self.Myb_r,self.Mzb_r))
        print('\n')
        
        self.Fzb_wdot = -((self.rho*self.Sw*self.Sh*self.lwt)/(np.pi*self.bw*self.bw))*self.CLw_a*self.CLh_a
        self.Myb_wdot = -self.xbh*self.Fzb_wdot
        self.Fxb_udot = 0.0
        self.Fxb_wdot = 0.0
        self.Fzb_udot = 0.0
        self.Myb_udot = 0.0
        
    def solve_derivatives2(self, dAlpha, dBeta, dp, dq, dr):
        
        u_o = self.eq_velo[0]
        v_o = self.eq_velo[1]
        w_o = self.eq_velo[2]
        
        p_o = (self.eq_rot[0])
        q_o = (self.eq_rot[1])
        r_o = (self.eq_rot[2])
        
        pbar_o = p_o*self.bw/(2.*self.V)
        qbar_o = q_o*self.cw/(2.*self.V)
        rbar_o = r_o*self.bw/(2.*self.V)
        
        # phi_o = self.eq_euler[0]
        # theta_o = self.eq_euler[1]
        
        tau_o = self.eq_inputs[0]
        alpha_o = (self.eq_inputs[1])
        beta_o = (self.eq_inputs[2])
        da_o = (self.eq_inputs[3])
        de_o = (self.eq_inputs[4])
        dr_o = (self.eq_inputs[5])
        
        FX0, FY0, FZ0, Mx0, My0, Mz0 = self.run_FM_body(np.array([alpha_o, beta_o, pbar_o, qbar_o, rbar_o, da_o, de_o, dr_o]), 0.0, self.V)
                
        '''dAlpha Data'''
        
        dAlpha_array = np.array([alpha_o - 2*dAlpha, alpha_o - dAlpha, alpha_o + dAlpha, alpha_o + 2*dAlpha])
        
        FM_dAlpha = np.zeros((4,6))
        
        for i in range(len(dAlpha_array)):
    
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([dAlpha_array[i], beta_o, pbar_o, qbar_o, rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            
            FM_dAlpha[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])
    
    
        dFX_dAlpha = self.force_derivs(FM_dAlpha[0,0],FM_dAlpha[1,0],FM_dAlpha[2,0],FM_dAlpha[3,0], dAlpha)
        dFY_dAlpha = self.force_derivs(FM_dAlpha[0,1],FM_dAlpha[1,1],FM_dAlpha[2,1],FM_dAlpha[3,1], dAlpha)
        dFZ_dAlpha = self.force_derivs(FM_dAlpha[0,2],FM_dAlpha[1,2],FM_dAlpha[2,2],FM_dAlpha[3,2], dAlpha)
            
        dMX_dAlpha = self.force_derivs(FM_dAlpha[0,3],FM_dAlpha[1,3],FM_dAlpha[2,3],FM_dAlpha[3,3], dAlpha)
        dMY_dAlpha = self.force_derivs(FM_dAlpha[0,4],FM_dAlpha[1,4],FM_dAlpha[2,4],FM_dAlpha[3,4], dAlpha)
        dMZ_dAlpha = self.force_derivs(FM_dAlpha[0,5],FM_dAlpha[1,5],FM_dAlpha[2,5],FM_dAlpha[3,5], dAlpha)
        
        '''dBeta Data'''
                
        dBeta_array = np.array([beta_o - 2*dBeta, beta_o - dBeta, beta_o + dBeta, beta_o + 2*dBeta])
    
        FM_dBeta = np.zeros((4,6))
        
        for i in range(len(dBeta_array)):
            
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha_o, dBeta_array[i], pbar_o, qbar_o, rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            
            FM_dBeta[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])
        
        dFX_dBeta = self.force_derivs(FM_dBeta[0,0],FM_dBeta[1,0],FM_dBeta[2,0],FM_dBeta[3,0], dBeta)
        dFY_dBeta = self.force_derivs(FM_dBeta[0,1],FM_dBeta[1,1],FM_dBeta[2,1],FM_dBeta[3,1], dBeta)
        dFZ_dBeta = self.force_derivs(FM_dBeta[0,2],FM_dBeta[1,2],FM_dBeta[2,2],FM_dBeta[3,2], dBeta)
            
        dMX_dBeta = self.force_derivs(FM_dBeta[0,3],FM_dBeta[1,3],FM_dBeta[2,3],FM_dBeta[3,3], dBeta)
        dMY_dBeta = self.force_derivs(FM_dBeta[0,4],FM_dBeta[1,4],FM_dBeta[2,4],FM_dBeta[3,4], dBeta)
        dMZ_dBeta = self.force_derivs(FM_dBeta[0,5],FM_dBeta[1,5],FM_dBeta[2,5],FM_dBeta[3,5], dBeta)
        
        
        '''Thrust Derivative'''
        
        # dT_dV = tau_o*self.dthrust_dV(self.V, self.rho, self.rho_0)
        dT_dV = self.thrust_derivatives(tau_o,self.V,self.H)
        
        # self.eq_FM is dimensional so it needs to be nondimensionalized and then
        # multiplied by the values
        dFX_dV = 2*FX0/self.V + dT_dV
        dFY_dV = 2*FY0/self.V
        dFZ_dV = 2*FZ0/self.V
        
        '''CHECK THIS'''
        dMX_dV = 2*Mx0/self.V
        dMY_dV = 2*My0/self.V
        dMZ_dV = 2*Mz0/self.V
        
        '''Could condense these calculations into arrays****'''
        
        '''derivative of body-fixed forces with respect to u'''
        self.Fxb_u = (u_o/self.V)*dFX_dV - (w_o/(u_o*u_o + w_o*w_o))*dFX_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFX_dBeta
        self.Fyb_u = (u_o/self.V)*dFY_dV - (w_o/(u_o*u_o + w_o*w_o))*dFY_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFY_dBeta
        self.Fzb_u = (u_o/self.V)*dFZ_dV - (w_o/(u_o*u_o + w_o*w_o))*dFZ_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFZ_dBeta
        
        '''derivative of body-fixed moments with respect to u'''
        self.Mxb_u = (u_o/self.V)*dMX_dV - (w_o/(u_o*u_o + w_o*w_o))*dMX_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMX_dBeta
        self.Myb_u = (u_o/self.V)*dMY_dV - (w_o/(u_o*u_o + w_o*w_o))*dMY_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMY_dBeta
        self.Mzb_u = (u_o/self.V)*dMZ_dV - (w_o/(u_o*u_o + w_o*w_o))*dMZ_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMZ_dBeta
        
        '''derivative of body-fixed forces with respect to v'''
        self.Fxb_v = (v_o/self.V)*dFX_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dFX_dBeta
        self.Fyb_v = (v_o/self.V)*dFY_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dFY_dBeta
        self.Fzb_v = (v_o/self.V)*dFZ_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dFZ_dBeta
        
        '''derivative of body-fixed moments with respect to v'''
        self.Mxb_v = (v_o/self.V)*dMX_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dMX_dBeta
        self.Myb_v = (v_o/self.V)*dMY_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dMY_dBeta
        self.Mzb_v = (v_o/self.V)*dMZ_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dMZ_dBeta
        
        '''derivative of body-fixed forces with respect to w'''
        self.Fxb_w = (w_o/self.V)*dFX_dV + (u_o/(u_o*u_o + w_o*w_o))*dFX_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFX_dBeta
        self.Fyb_w = (w_o/self.V)*dFY_dV + (u_o/(u_o*u_o + w_o*w_o))*dFY_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFY_dBeta
        self.Fzb_w = (w_o/self.V)*dFZ_dV + (u_o/(u_o*u_o + w_o*w_o))*dFZ_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFZ_dBeta
        
        '''derivative of body-fixed moments with respect to w'''
        self.Mxb_w = (w_o/self.V)*dMX_dV + (u_o/(u_o*u_o + w_o*w_o))*dMX_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMX_dBeta
        self.Myb_w = (w_o/self.V)*dMY_dV + (u_o/(u_o*u_o + w_o*w_o))*dMY_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMY_dBeta
        self.Mzb_w = (w_o/self.V)*dMZ_dV + (u_o/(u_o*u_o + w_o*w_o))*dMZ_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMZ_dBeta
        
        
        '''delta rotation rate Data'''
        dpbar = dp*self.bw/(2.*self.V)
        dqbar = dq*self.cw/(2.*self.V)
        drbar = dr*self.bw/(2.*self.V)
        
        dPbar_array = np.array([pbar_o - 2*dpbar, pbar_o - dpbar, pbar_o + dpbar, pbar_o + 2*dpbar])
        dQbar_array = np.array([qbar_o - 2*dqbar, qbar_o - dqbar, qbar_o + dqbar, qbar_o + 2*dqbar])
        dRbar_array = np.array([rbar_o - 2*drbar, rbar_o - drbar, rbar_o + drbar, rbar_o + 2*drbar])
        
        FM_dPbar = np.zeros((4,6))
        FM_dQbar = np.zeros((4,6))
        FM_dRbar = np.zeros((4,6))
        
        for i in range(len(dPbar_array)):
            
            FXp, FYp, FZp, Mxp, Myp, Mzp = self.run_FM_body(np.array([alpha_o, beta_o, dPbar_array[i], qbar_o, rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            FM_dPbar[i,:] = np.array([FXp, FYp, FZp, Mxp, Myp, Mzp])
            
            FXq, FYq, FZq, Mxq, Myq, Mzq = self.run_FM_body(np.array([alpha_o, beta_o, pbar_o, dQbar_array[i], rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            FM_dQbar[i,:] = np.array([FXq, FYq, FZq, Mxq, Myq, Mzq])
            
            FXr, FYr, FZr, Mxr, Myr, Mzr = self.run_FM_body(np.array([alpha_o, beta_o, pbar_o, qbar_o, dRbar_array[i], da_o, de_o, dr_o]), tau_o, self.V) 
            FM_dRbar[i,:] = np.array([FXr, FYr, FZr, Mxr, Myr, Mzr])
        
        self.Fxb_p = self.force_derivs(FM_dPbar[0,0],FM_dPbar[1,0],FM_dPbar[2,0],FM_dPbar[3,0], dp)
        self.Fyb_p = self.force_derivs(FM_dPbar[0,1],FM_dPbar[1,1],FM_dPbar[2,1],FM_dPbar[3,1], dp)
        self.Fzb_p = self.force_derivs(FM_dPbar[0,2],FM_dPbar[1,2],FM_dPbar[2,2],FM_dPbar[3,2], dp)
            
        self.Mxb_p = self.force_derivs(FM_dPbar[0,3],FM_dPbar[1,3],FM_dPbar[2,3],FM_dPbar[3,3], dp)
        self.Myb_p = self.force_derivs(FM_dPbar[0,4],FM_dPbar[1,4],FM_dPbar[2,4],FM_dPbar[3,4], dp)
        self.Mzb_p = self.force_derivs(FM_dPbar[0,5],FM_dPbar[1,5],FM_dPbar[2,5],FM_dPbar[3,5], dp)
        
        self.Fxb_q = self.force_derivs(FM_dQbar[0,0],FM_dQbar[1,0],FM_dQbar[2,0],FM_dQbar[3,0], dq)
        self.Fyb_q = self.force_derivs(FM_dQbar[0,1],FM_dQbar[1,1],FM_dQbar[2,1],FM_dQbar[3,1], dq)
        self.Fzb_q = self.force_derivs(FM_dQbar[0,2],FM_dQbar[1,2],FM_dQbar[2,2],FM_dQbar[3,2], dq)
            
        self.Mxb_q = self.force_derivs(FM_dQbar[0,3],FM_dQbar[1,3],FM_dQbar[2,3],FM_dQbar[3,3], dq)
        self.Myb_q = self.force_derivs(FM_dQbar[0,4],FM_dQbar[1,4],FM_dQbar[2,4],FM_dQbar[3,4], dq)
        self.Mzb_q = self.force_derivs(FM_dQbar[0,5],FM_dQbar[1,5],FM_dQbar[2,5],FM_dQbar[3,5], dq)
        
        self.Fxb_r = self.force_derivs(FM_dRbar[0,0],FM_dRbar[1,0],FM_dRbar[2,0],FM_dRbar[3,0], dr)
        self.Fyb_r = self.force_derivs(FM_dRbar[0,1],FM_dRbar[1,1],FM_dRbar[2,1],FM_dRbar[3,1], dr)
        self.Fzb_r = self.force_derivs(FM_dRbar[0,2],FM_dRbar[1,2],FM_dRbar[2,2],FM_dRbar[3,2], dr)
            
        self.Mxb_r = self.force_derivs(FM_dRbar[0,3],FM_dRbar[1,3],FM_dRbar[2,3],FM_dRbar[3,3], dr)
        self.Myb_r = self.force_derivs(FM_dRbar[0,4],FM_dRbar[1,4],FM_dRbar[2,4],FM_dRbar[3,4], dr)
        self.Mzb_r = self.force_derivs(FM_dRbar[0,5],FM_dRbar[1,5],FM_dRbar[2,5],FM_dRbar[3,5], dr)
        
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,u derivatvies:',self.Fxb_u,self.Fyb_u,self.Fzb_u,self.Mxb_u,self.Myb_u,self.Mzb_u))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,v derivatvies:',self.Fxb_v,self.Fyb_v,self.Fzb_v,self.Mxb_v,self.Myb_v,self.Mzb_v))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,w derivatvies:',self.Fxb_w,self.Fyb_w,self.Fzb_w,self.Mxb_w,self.Myb_w,self.Mzb_w))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,p derivatvies:',self.Fxb_p,self.Fyb_p,self.Fzb_p,self.Mxb_p,self.Myb_p,self.Mzb_p))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,q derivatvies:',self.Fxb_q,self.Fyb_q,self.Fzb_q,self.Mxb_q,self.Myb_q,self.Mzb_q))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,r derivatvies:',self.Fxb_r,self.Fyb_r,self.Fzb_r,self.Mxb_r,self.Myb_r,self.Mzb_r))
        print('\n')
        
        self.acceleration_derivatives()
        
        # print('dFXdalpha: ', dFX_dAlpha)
        # print('dFXdbeta: ', dFX_dBeta)
        # print('dTdV: ', dT_dV)
        print('\n')
        
    def solve_derivatives_austin(self, duw, dvw, dww, dpw, dqw, drw):
        
        u_o = self.eq_velo[0]
        v_o = self.eq_velo[1]
        w_o = self.eq_velo[2]
        
        V_o = np.sqrt(u_o*u_o + v_o*v_o + w_o*w_o)
        
        p_o = (self.eq_rot[0])
        q_o = (self.eq_rot[1])
        r_o = (self.eq_rot[2])
                
        # phi_o = self.eq_euler[0]
        # theta_o = self.eq_euler[1]
        
        tau_o = self.eq_inputs[0]
        alpha_o = (self.eq_inputs[1])
        beta_o = (self.eq_inputs[2])
        da_o = (self.eq_inputs[3])
        de_o = (self.eq_inputs[4])
        dr_o = (self.eq_inputs[5])
        
        pbar_o = p_o*self.bw/(2.*V_o)
        qbar_o = q_o*self.cw/(2.*V_o)
        rbar_o = r_o*self.bw/(2.*V_o)

        dub_vec = self.wind_2_body_vector(np.array([duw,0.0,0.0]),alpha_o,beta_o)
        dvb_vec = self.wind_2_body_vector(np.array([0.0, dvw,0.0]),alpha_o,beta_o)
        dwb_vec = self.wind_2_body_vector(np.array([0.0,0.0,dww]),alpha_o,beta_o)
        dpb_vec = self.wind_2_body_vector(np.array([dpw,0.0,0.0]),self.alpha,self.beta)
        dqb_vec = self.wind_2_body_vector(np.array([0.0, dqw,0.0]),self.alpha,self.beta)
        drb_vec = self.wind_2_body_vector(np.array([0.0,0.0,drw]),self.alpha,self.beta)
        
        
        # du derivatives
        dub_matrix = np.array([[u_o - 2*dub_vec[0], v_o - 2*dub_vec[1], w_o - 2*dub_vec[2]],
                               [u_o - dub_vec[0], v_o - dub_vec[1], w_o - dub_vec[2]],
                               [u_o + dub_vec[0], v_o + dub_vec[1], w_o + dub_vec[2]],
                               [u_o + 2*dub_vec[0], v_o + 2*dub_vec[1], w_o + 2*dub_vec[2]]])
        
        FM_dub = np.zeros((4,6))
        
        for i in range(4):
            V = np.sqrt(dub_matrix[i,0]*dub_matrix[i,0] + dub_matrix[i,1]*dub_matrix[i,1] + dub_matrix[i,2]*dub_matrix[i,2])
            
            pbar = p_o*self.bw/(2.*V)
            qbar = q_o*self.cw/(2.*V)
            rbar = r_o*self.bw/(2.*V)
            
            alpha = np.arctan2(dub_matrix[i,2], dub_matrix[i,0])
            beta = np.arcsin(dub_matrix[i,1]/V)
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha, beta, pbar, qbar, rbar, da_o, de_o, dr_o]), tau_o, V) 
            
            FM_dub[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])


        self.Fxb_u = self.force_derivs(FM_dub[0,0],FM_dub[1,0],FM_dub[2,0],FM_dub[3,0], du)
        self.Fyb_u = self.force_derivs(FM_dub[0,1],FM_dub[1,1],FM_dub[2,1],FM_dub[3,1], du)
        self.Fzb_u = self.force_derivs(FM_dub[0,2],FM_dub[1,2],FM_dub[2,2],FM_dub[3,2], du)     
        self.Mxb_u = self.force_derivs(FM_dub[0,3],FM_dub[1,3],FM_dub[2,3],FM_dub[3,3], du)
        self.Myb_u = self.force_derivs(FM_dub[0,4],FM_dub[1,4],FM_dub[2,4],FM_dub[3,4], du)
        self.Mzb_u = self.force_derivs(FM_dub[0,5],FM_dub[1,5],FM_dub[2,5],FM_dub[3,5], du)
        
        
        # dv derivatives
        dvb_matrix = np.array([[u_o - 2*dvb_vec[0], v_o - 2*dvb_vec[1], w_o - 2*dvb_vec[2]],
                               [u_o - dvb_vec[0], v_o - dvb_vec[1], w_o - dvb_vec[2]],
                               [u_o + dvb_vec[0], v_o + dvb_vec[1], w_o + dvb_vec[2]],
                               [u_o + 2*dvb_vec[0], v_o + 2*dvb_vec[1], w_o + 2*dvb_vec[2]]])
        
        FM_dvb = np.zeros((4,6))
        
        for i in range(4):
            V = np.sqrt(dvb_matrix[i,0]*dvb_matrix[i,0] + dvb_matrix[i,1]*dvb_matrix[i,1] + dvb_matrix[i,2]*dvb_matrix[i,2])
            
            pbar = p_o*self.bw/(2.*V)
            qbar = q_o*self.cw/(2.*V)
            rbar = r_o*self.bw/(2.*V)
            
            alpha = np.arctan2(dvb_matrix[i,2], dvb_matrix[i,0])
            beta = np.arcsin(dvb_matrix[i,1]/V)
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha, beta, pbar, qbar, rbar, da_o, de_o, dr_o]), tau_o, V) 
            
            FM_dvb[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])


        self.Fxb_v = self.force_derivs(FM_dvb[0,0],FM_dvb[1,0],FM_dvb[2,0],FM_dvb[3,0], dv)
        self.Fyb_v = self.force_derivs(FM_dvb[0,1],FM_dvb[1,1],FM_dvb[2,1],FM_dvb[3,1], dv)
        self.Fzb_v = self.force_derivs(FM_dvb[0,2],FM_dvb[1,2],FM_dvb[2,2],FM_dvb[3,2], dv)     
        self.Mxb_v = self.force_derivs(FM_dvb[0,3],FM_dvb[1,3],FM_dvb[2,3],FM_dvb[3,3], dv)
        self.Myb_v = self.force_derivs(FM_dvb[0,4],FM_dvb[1,4],FM_dvb[2,4],FM_dvb[3,4], dv)
        self.Mzb_v = self.force_derivs(FM_dvb[0,5],FM_dvb[1,5],FM_dvb[2,5],FM_dvb[3,5], dv)
        
        # dw derivatives
        dwb_matrix = np.array([[u_o - 2*dwb_vec[0], v_o - 2*dwb_vec[1], w_o - 2*dwb_vec[2]],
                               [u_o - dwb_vec[0], v_o - dwb_vec[1], w_o - dwb_vec[2]],
                               [u_o + dwb_vec[0], v_o + dwb_vec[1], w_o + dwb_vec[2]],
                               [u_o + 2*dwb_vec[0], v_o + 2*dwb_vec[1], w_o + 2*dwb_vec[2]]])
        
        FM_dwb = np.zeros((4,6))
        
        for i in range(4):
            V = np.sqrt(dwb_matrix[i,0]*dwb_matrix[i,0] + dwb_matrix[i,1]*dwb_matrix[i,1] + dwb_matrix[i,2]*dwb_matrix[i,2])
            
            pbar = p_o*self.bw/(2.*V)
            qbar = q_o*self.cw/(2.*V)
            rbar = r_o*self.bw/(2.*V)
            
            alpha = np.arctan2(dwb_matrix[i,2], dwb_matrix[i,0])
            beta = np.arcsin(dwb_matrix[i,1]/V)
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha, beta, pbar, qbar, rbar, da_o, de_o, dr_o]), tau_o, V) 
            
            FM_dwb[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])


        self.Fxb_w = self.force_derivs(FM_dwb[0,0],FM_dwb[1,0],FM_dwb[2,0],FM_dwb[3,0], dw)
        self.Fyb_w = self.force_derivs(FM_dwb[0,1],FM_dwb[1,1],FM_dwb[2,1],FM_dwb[3,1], dw)
        self.Fzb_w = self.force_derivs(FM_dwb[0,2],FM_dwb[1,2],FM_dwb[2,2],FM_dwb[3,2], dw)     
        self.Mxb_w = self.force_derivs(FM_dwb[0,3],FM_dwb[1,3],FM_dwb[2,3],FM_dwb[3,3], dw)
        self.Myb_w = self.force_derivs(FM_dwb[0,4],FM_dwb[1,4],FM_dwb[2,4],FM_dwb[3,4], dw)
        self.Mzb_w = self.force_derivs(FM_dwb[0,5],FM_dwb[1,5],FM_dwb[2,5],FM_dwb[3,5], dw)
        
        # dp derivatives
        dpb_matrix = np.array([[p_o - 2*dpb_vec[0], q_o - 2*dpb_vec[1], r_o - 2*dpb_vec[2]],
                               [p_o - dpb_vec[0], q_o - dpb_vec[1], r_o - dpb_vec[2]],
                               [p_o + dpb_vec[0], q_o + dpb_vec[1], r_o + dpb_vec[2]],
                               [p_o + 2*dpb_vec[0], q_o + 2*dpb_vec[1], r_o + 2*dpb_vec[2]]])
        
        FM_dpb = np.zeros((4,6))
        
        for i in range(4):
            
            pbar = dpb_matrix[i,0]*self.bw/(2.*V_o)
            qbar = dpb_matrix[i,1]*self.cw/(2.*V_o)
            rbar = dpb_matrix[i,2]*self.bw/(2.*V_o)
            
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha_o, beta_o, pbar, qbar, rbar, da_o, de_o, dr_o]), tau_o, V) 
            
            FM_dpb[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])


        self.Fxb_p = self.force_derivs(FM_dpb[0,0],FM_dpb[1,0],FM_dpb[2,0],FM_dpb[3,0], dp)
        self.Fyb_p = self.force_derivs(FM_dpb[0,1],FM_dpb[1,1],FM_dpb[2,1],FM_dpb[3,1], dp)
        self.Fzb_p = self.force_derivs(FM_dpb[0,2],FM_dpb[1,2],FM_dpb[2,2],FM_dpb[3,2], dp)     
        self.Mxb_p = self.force_derivs(FM_dpb[0,3],FM_dpb[1,3],FM_dpb[2,3],FM_dpb[3,3], dp)
        self.Myb_p = self.force_derivs(FM_dpb[0,4],FM_dpb[1,4],FM_dpb[2,4],FM_dpb[3,4], dp)
        self.Mzb_p = self.force_derivs(FM_dpb[0,5],FM_dpb[1,5],FM_dpb[2,5],FM_dpb[3,5], dp)
        
        # dq derivatives
        dqb_matrix = np.array([[p_o - 2*dqb_vec[0], q_o - 2*dqb_vec[1], r_o - 2*dqb_vec[2]],
                               [p_o - dqb_vec[0], q_o - dqb_vec[1], r_o - dqb_vec[2]],
                               [p_o + dqb_vec[0], q_o + dqb_vec[1], r_o + dqb_vec[2]],
                               [p_o + 2*dqb_vec[0], q_o + 2*dqb_vec[1], r_o + 2*dqb_vec[2]]])
        
        FM_dqb = np.zeros((4,6))
        
        for i in range(4):
            
            pbar = dqb_matrix[i,0]*self.bw/(2.*V_o)
            qbar = dqb_matrix[i,1]*self.cw/(2.*V_o)
            rbar = dqb_matrix[i,2]*self.bw/(2.*V_o)
            
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha_o, beta_o, pbar, qbar, rbar, da_o, de_o, dr_o]), tau_o, V) 
            
            FM_dqb[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])


        self.Fxb_q = self.force_derivs(FM_dqb[0,0],FM_dqb[1,0],FM_dqb[2,0],FM_dqb[3,0], dq)
        self.Fyb_q = self.force_derivs(FM_dqb[0,1],FM_dqb[1,1],FM_dqb[2,1],FM_dqb[3,1], dq)
        self.Fzb_q = self.force_derivs(FM_dqb[0,2],FM_dqb[1,2],FM_dqb[2,2],FM_dqb[3,2], dq)     
        self.Mxb_q = self.force_derivs(FM_dqb[0,3],FM_dqb[1,3],FM_dqb[2,3],FM_dqb[3,3], dq)
        self.Myb_q = self.force_derivs(FM_dqb[0,4],FM_dqb[1,4],FM_dqb[2,4],FM_dqb[3,4], dq)
        self.Mzb_q = self.force_derivs(FM_dqb[0,5],FM_dqb[1,5],FM_dqb[2,5],FM_dqb[3,5], dq)
        
        # dr derivatives
        drb_matrix = np.array([[p_o - 2*drb_vec[0], q_o - 2*drb_vec[1], r_o - 2*drb_vec[2]],
                               [p_o - drb_vec[0], q_o - drb_vec[1], r_o - drb_vec[2]],
                               [p_o + drb_vec[0], q_o + drb_vec[1], r_o + drb_vec[2]],
                               [p_o + 2*drb_vec[0], q_o + 2*drb_vec[1], r_o + 2*drb_vec[2]]])
        
        FM_drb = np.zeros((4,6))
        
        for i in range(4):
            
            pbar = drb_matrix[i,0]*self.bw/(2.*V_o)
            qbar = drb_matrix[i,1]*self.cw/(2.*V_o)
            rbar = drb_matrix[i,2]*self.bw/(2.*V_o)
            
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha_o, beta_o, pbar, qbar, rbar, da_o, de_o, dr_o]), tau_o, V) 
            
            FM_drb[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])


        self.Fxb_r = self.force_derivs(FM_drb[0,0],FM_drb[1,0],FM_drb[2,0],FM_drb[3,0], dr)
        self.Fyb_r = self.force_derivs(FM_drb[0,1],FM_drb[1,1],FM_drb[2,1],FM_drb[3,1], dr)
        self.Fzb_r = self.force_derivs(FM_drb[0,2],FM_drb[1,2],FM_drb[2,2],FM_drb[3,2], dr)     
        self.Mxb_r = self.force_derivs(FM_drb[0,3],FM_drb[1,3],FM_drb[2,3],FM_drb[3,3], dr)
        self.Myb_r = self.force_derivs(FM_drb[0,4],FM_drb[1,4],FM_drb[2,4],FM_drb[3,4], dr)
        self.Mzb_r = self.force_derivs(FM_drb[0,5],FM_drb[1,5],FM_drb[2,5],FM_drb[3,5], dr)
        
        # print('{:<16}{:<20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,u derivatvies:',self.Fxb_u,self.Fyb_u,self.Fzb_u,self.Mxb_u,self.Myb_u,self.Mzb_u))
        # print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,v derivatvies:',self.Fxb_v,self.Fyb_v,self.Fzb_v,self.Mxb_v,self.Myb_v,self.Mzb_v))
        # print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,w derivatvies:',self.Fxb_w,self.Fyb_w,self.Fzb_w,self.Mxb_w,self.Myb_w,self.Mzb_w))
        # print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,p derivatvies:',self.Fxb_p,self.Fyb_p,self.Fzb_p,self.Mxb_p,self.Myb_p,self.Mzb_p))
        # print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,q derivatvies:',self.Fxb_q,self.Fyb_q,self.Fzb_q,self.Mxb_q,self.Myb_q,self.Mzb_q))
        # print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,r derivatvies:',self.Fxb_r,self.Fyb_r,self.Fzb_r,self.Mxb_r,self.Myb_r,self.Mzb_r))
        # print('\n')
        
        
        self.Fzb_wdot = -((self.rho*self.Sw*self.Sh*self.lwt)/(np.pi*self.bw*self.bw))*self.CLw_a*self.CLh_a
        self.Myb_wdot = -self.xbh*self.Fzb_wdot
        self.Fxb_udot = 0.0
        self.Fxb_wdot = 0.0
        self.Fzb_udot = 0.0
        self.Myb_udot = 0.0
    
    def run_FM_body(self, params, tau, V):
        
        '''Generate body-fixed force and moments for the flight condition defined
        with the input parameters and throttle setting. Uses the aeromodels
        to generate coefficients in the wind coordinate system and then converts to 
        and returns the body-fixed forces and moments.'''
        Mach = V/self.a
        
        CL, CS, CD, Cl, Cm, Cn = self.aeroModel.aero_results(*params, compressible = self.compressible, M = Mach, enforce_stall=self.stall)
        #convert forces to body-fixed coordinates
        CX, CY, CZ = self.wind_2_body_forces([CL, CS, CD], alpha = params[0], beta = params[1])
        # thrust at equilibrium airspeed
        FX_thrust = tau*self.thrust(V, self.rho, self.rho_0)
        CFMw = np.array([CX, CY, CZ, Cl, Cm, Cn])
        # redimensionalize forces and moments
        FX, FY, FZ, Mx, My, Mz = self.redim_FM(self.rho, V, self.Sw, self.bw, self.cw, CFMw, FX_thrust, self.cg_shift)
        
    def wind_2_body_forces(self, forces, alpha, beta):
        
        CL, CS, CD = forces
        
        CXb = -(CD*np.cos(alpha)*np.cos(beta) + CS*np.cos(alpha)*np.sin(beta) - CL*np.sin(alpha))
        CYb = CS*np.cos(beta) - CD*np.sin(beta)
        CZb = -(CD*np.sin(alpha)*np.cos(beta) + CS*np.sin(alpha)*np.sin(beta) + CL*np.cos(alpha))
            
        return np.array([CXb, CYb, CZb])

    def wind_2_body_vector(self, vector, alpha, beta):
        
        x, y, z = vector
        
        Xb = (x*np.cos(alpha)*np.cos(beta) - y*np.cos(alpha)*np.sin(beta) - z*np.sin(alpha))
        Yb = y*np.cos(beta) + x*np.sin(beta)
        Zb = (x*np.sin(alpha)*np.cos(beta) - y*np.sin(alpha)*np.sin(beta) + z*np.cos(alpha))
            
        return np.array([Xb, Yb, Zb])


    def redim_FM(self, rho, V, Sw, bw, cw, CFMw, FX_thrust, cg_shift):
        '''Redimensionalize force and moment coefficients. Adds thrust force
        and changes to body-fixed moments because of CG shift.'''
        
        redim_const = 0.5*rho*V*V*Sw
        FX = CFMw[0]*redim_const
        FY = CFMw[1]*redim_const
        FZ = CFMw[2]*redim_const
        FX = FX + FX_thrust
        # shift forces and moments to be around the CG shift location
        # redimensionalize moments
        MX = CFMw[3]*redim_const*bw - FZ*cg_shift[1] + FY*cg_shift[2]
        MY = CFMw[4]*redim_const*cw - FX*cg_shift[2] + FZ*cg_shift[0]
        MZ = CFMw[5]*redim_const*bw - FY*cg_shift[0] + FX*cg_shift[1]
        
        return FX, FY, FZ, MX, MY, MZ

    def force_derivs(self, Fm2, Fm1, Fp1, Fp2, delta):
        '''fourth order central difference'''
        df_prime = (-Fp2 + 8*Fp1 - 8*Fm1 + Fm2)/(12*delta)
        return df_prime
    
    def thrust(self, V, rho, rho_0):
        # thrust model for the F-16/BIRE as a function of velocity
        T0 = 29550.
        T1 = 0.
        T2 = 0.
        a = 0.84
        C1 = (rho/rho_0)**a
        C2 = T0 + T1*V + T2*V**2
        T = C1*C2
        return T

    def dthrust_dV(self, V, rho, rho_0):
        # thrust model for the F-16/BIRE as a function of velocity
        T0 = 29550.
        T1 = 0.
        T2 = 0.
        a = 0.84
        C1 = (rho/rho_0)**a
        C2 = T1 + 2*T2*V
        dT = C1*C2
        return dT
    
    def find_body_velocity(self, alpha, beta):
        '''Find body-fixed velocities from alpha, beta, total airspeed'''
        self.alpha = alpha
        self.beta = beta
        u = self.V*np.cos(self.alpha)*np.cos(self.beta)
        v = self.V*np.sin(self.beta)
        w = self.V*np.sin(self.alpha)*np.cos(self.beta)
        
        self.eq_velo = np.array([u, v, w])

    def get_JSON_inputs(self, filename):
        
        json_data = json.loads(open(filename).read()) # reads json input fil
        
        self.g = json_data["operating"]["g[ft/s^2]"]
        
        # AIRCRAFT INPUTS
        self.W = json_data["operating"]["weight[lbf]"]
        self.Sw = json_data["aircraft"]["wing_area[ft^2]"] #ft^2
        self.bw = json_data["aircraft"]["wing_span[ft]"] #ft
        self.cw = self.Sw/self.bw
        
        self.Ixxb = json_data["reference"]["Ixx[slugs*ft^2]"]
        self.Iyyb = json_data["reference"]["Iyy[slugs*ft^2]"]
        self.Izzb = json_data["reference"]["Izz[slugs*ft^2]"]
        self.Ixyb = json_data["reference"]["Ixy[slugs*ft^2]"]
        self.Ixzb = json_data["reference"]["Ixz[slugs*ft^2]"]
        self.Iyzb = json_data["reference"]["Iyz[slugs*ft^2]"]
        
        self.hxb = json_data["reference"]["hx[slug*ft^2/s]"]
        self.hyb = json_data["reference"]["hy[slug*ft^2/s]"]
        self.hzb = json_data["reference"]["hz[slug*ft^2/s]"]

        
        self.V = json_data["operating"]["airspeed[ft/s]"]
        
        theta = np.deg2rad(json_data["operating"]["elevation_angle[deg]"])
        phi = np.deg2rad(json_data["operating"]["bank_angle[deg]"])
        
        self.eq_euler = np.array([phi,theta])
        
        self.rho = json_data["operating"]["density[slugs/ft^3]"]
        
        redim_coeff = 0.5*self.rho*self.V*self.V*self.Sw
        
        CXo = json_data["reference"]["CXo"]
        CZo = json_data["reference"]["CZo"]
        Cmo = json_data["reference"]["Cmo"]
        
        self.Fxb_u = (1/self.V)*redim_coeff*(2*CXo + json_data["reference"]["CX,mu"])
        self.Fyb_u = (1/self.V)*redim_coeff*json_data["reference"]["CY,mu"]
        self.Fzb_u = (1/self.V)*redim_coeff*(2*CZo + json_data["reference"]["CZ,mu"])
        self.Mxb_u = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cl,mu"]
        self.Myb_u = (1/self.V)*self.cw*redim_coeff*(2*Cmo + json_data["reference"]["Cm,mu"])
        self.Mzb_u = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cn,mu"]
        
        self.Fxb_v = (1/self.V)*redim_coeff*json_data["reference"]["CX,beta"]
        self.Fyb_v = (1/self.V)*redim_coeff*json_data["reference"]["CY,beta"]
        self.Fzb_v = (1/self.V)*redim_coeff*json_data["reference"]["CZ,beta"]
        self.Mxb_v = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cl,beta"]
        self.Myb_v = (1/self.V)*self.cw*redim_coeff*json_data["reference"]["Cm,beta"]
        self.Mzb_v = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cn,beta"]
        
        self.Fxb_w = (1/self.V)*redim_coeff*json_data["reference"]["CX,alpha"]
        self.Fyb_w = (1/self.V)*redim_coeff*json_data["reference"]["CY,alpha"]
        self.Fzb_w = (1/self.V)*redim_coeff*json_data["reference"]["CZ,alpha"]
        self.Mxb_w = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cl,alpha"]
        self.Myb_w = (1/self.V)*self.cw*redim_coeff*json_data["reference"]["Cm,alpha"]
        self.Mzb_w = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cn,alpha"]
        
        self.Fxb_wdot = (self.cw/(2*self.V*self.V))*redim_coeff*json_data["reference"]["CX,alphahat"]
        self.Fzb_wdot = (self.cw/(2*self.V*self.V))*redim_coeff*json_data["reference"]["CZ,alphahat"]
        self.Myb_wdot = (self.cw/(2*self.V*self.V))*redim_coeff*self.cw*json_data["reference"]["Cm,alphahat"]
        
        self.Fxb_udot = (self.cw/(2*self.V*self.V))*redim_coeff*json_data["reference"]["CX,muhat"]
        self.Fzb_udot = (self.cw/(2*self.V*self.V))*redim_coeff*json_data["reference"]["CZ,muhat"]
        self.Myb_udot = (self.cw/(2*self.V*self.V))*redim_coeff*self.cw*json_data["reference"]["Cm,muhat"]

        
        self.Fxb_p = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CX,pbar"]
        self.Fyb_p = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CY,pbar"]
        self.Fzb_p = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CZ,pbar"]
        self.Mxb_p = (self.bw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cl,pbar"]
        self.Myb_p = (self.bw/(2*self.V))*self.cw*redim_coeff*json_data["reference"]["Cm,pbar"]
        self.Mzb_p = (self.bw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cn,pbar"]
        
        self.Fxb_q = (self.cw/(2*self.V))*redim_coeff*json_data["reference"]["CX,qbar"]
        self.Fyb_q = (self.cw/(2*self.V))*redim_coeff*json_data["reference"]["CY,qbar"]
        self.Fzb_q = (self.cw/(2*self.V))*redim_coeff*json_data["reference"]["CZ,qbar"]
        self.Mxb_q = (self.cw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cl,qbar"]
        self.Myb_q = (self.cw/(2*self.V))*self.cw*redim_coeff*json_data["reference"]["Cm,qbar"]
        self.Mzb_q = (self.cw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cn,qbar"]
        
        self.Fxb_r = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CX,rbar"]
        self.Fyb_r = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CY,rbar"]
        self.Fzb_r = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CZ,rbar"]
        self.Mxb_r = (self.bw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cl,rbar"]
        self.Myb_r = (self.bw/(2*self.V))*self.cw*redim_coeff*json_data["reference"]["Cm,rbar"]
        self.Mzb_r = (self.bw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cn,rbar"]
     