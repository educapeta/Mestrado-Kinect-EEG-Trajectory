def _paint_mi_animation(self, painter, width, height):
        """Animacao da MI baseada na trajetoria REAL da ME.

        Projeta os pontos 3D em 2D (normalizando bounds, ignorando outliers/saltos).
        Desenha: mesa, home, alvo, caminho (linha continua), marcador movel (ponto),
        objeto acompanhando nas fases de levantamento/retorno.
        Fallback para esquema generico se nao houver trajetoria valida.
        """
        # Get trajectory data
        traj = getattr(self, "_mi_anim_traj", None)
        if not traj or len(traj) < 2:
            # Fallback: draw generic 4-phase animation like _paint_animation
            self._paint_animation(painter, width, height)
            return

        # Extract positions (ignore times, use only 3D positions)
        positions = []
        for t, pos in traj:
            if pos is not None and np.isfinite(pos).all():
                positions.append(np.asarray(pos, np.float64))
        
        if len(positions) < 2:
            self._paint_animation(painter, width, height)
            return

        positions = np.array(positions)  # (n, 3)
        
        # Normalize bounds: center and scale to fit in view
        # Remove outliers (points beyond 3 std from median)
        med = np.median(positions, axis=0)
        std = np.std(positions, axis=0)
        mask = np.all(np.abs(positions - med) <= 3 * std, axis=1)
        if np.sum(mask) < 2:
            self._paint_animation(painter, width, height)
            return
        positions = positions[mask]
        
        # Project to 2D: use X (left-right) and Z (depth) as horizontal/vertical
        # Y is height (up-down), we'll use it for lift visualization
        x_vals = positions[:, 0]
        z_vals = positions[:, 2]
        y_vals = positions[:, 1]
        
        # Normalize to [0, 1] range
        x_min, x_max = np.min(x_vals), np.max(x_vals)
        z_min, z_max = np.min(z_vals), np.max(z_vals)
        y_min, y_max = np.min(y_vals), np.max(y_vals)
        
        # Avoid division by zero
        x_range = max(x_max - x_min, 1e-3)
        z_range = max(z_max - z_min, 1e-3)
        y_range = max(y_max - y_min, 1e-3)
        
        # Map to screen coordinates (table area)
        # Table area: left=6%, right=94%, top=20%, bottom=92% of screen
        table_left = width * 0.06
        table_right = width * 0.94
        table_top = height * 0.20
        table_bottom = height * 0.92
        table_w = table_right - table_left
        table_h = table_bottom - table_top
        
        # Home position (center bottom of table)
        home_x = width * 0.5
        home_y = height * 0.80
        
        # Target position (object side)
        lado = OBJECT_SIDE.get(self.object_name, "direita")
        alvo_x = width * (0.76 if lado == "direita" else 0.24)
        alvo_y = height * 0.38
        
        # Convert trajectory to screen coordinates
        screen_points = []
        for i in range(len(positions)):
            # Normalize X to table width (relative to home)
            norm_x = (x_vals[i] - x_min) / x_range
            # Normalize Z to table depth (relative to home at back)
            norm_z = 1.0 - (z_vals[i] - z_min) / z_range  # flip so depth goes up
            # Screen position
            sx = table_left + norm_x * table_w
            sy = table_top + norm_z * table_h
            screen_points.append(QPointF(sx, sy))
        
        # Draw table
        esp = max(2.0, min(width, height) * 0.004)
        mesa = QRectF(table_left, table_top, table_w, table_h)
        painter.setPen(QPen(self.TABLE, esp))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(mesa, 18, 18)
        
        # Draw full trajectory path (dashed)
        path_full = QPainterPath()
        if screen_points:
            path_full.moveTo(screen_points[0])
            for pt in screen_points[1:]:
                path_full.lineTo(pt)
        
        painter.setPen(QPen(self.PATH, esp * 1.2, Qt.DashLine, Qt.RoundCap))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path_full)
        
        # Animation progress
        duracao = max(1e-3, float(getattr(self, "_mi_anim_dur", MI_SEC)))
        decorrido = time.monotonic() - getattr(self, "_mi_anim_t0", time.monotonic())
        progresso = min(1.0, max(0.0, decorrido / duracao))
        suave = progresso * progresso * (3.0 - 2.0 * progresso)  # smoothstep
        
        # Draw partial path (solid) up to current progress
        n_pts = len(screen_points)
        if n_pts >= 2:
            idx = max(1, min(n_pts - 1, int(suave * (n_pts - 1))))
            path_partial = QPainterPath()
            path_partial.moveTo(screen_points[0])
            for i in range(1, idx + 1):
                path_partial.lineTo(screen_points[i])
            painter.setPen(QPen(self.WHITE, esp * 1.8, Qt.SolidLine, Qt.RoundCap))
            painter.drawPath(path_partial)
            
            # Current position marker
            current_pt = screen_points[idx]
            raio_ponto = max(8.0, min(width, height) * 0.014)
            painter.setPen(QPen(QColor(20, 20, 22), esp))
            painter.setBrush(self.WHITE)
            painter.drawEllipse(current_pt, raio_ponto, raio_ponto)
        
        # Draw target (alvo)
        raio = max(24.0, min(width, height) * 0.05)
        painter.setPen(QPen(self.TARGET, esp * 1.4))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(alvo_x, alvo_y), raio, raio)
        
        # Object follows hand during transport (approximate: middle 40% of trajectory)
        em_transporte = 0.30 <= suave <= 0.70
        pos_obj = QPointF(current_pt) if (em_transporte and n_pts >= 2) else QPointF(alvo_x, alvo_y)
        caixa = QRectF(pos_obj.x() - raio * 0.62,
                       pos_obj.y() - raio * 0.62,
                       raio * 1.24, raio * 1.24)
        painter.setPen(QPen(self.WHITE, max(4.0, esp * 2.0)))
        painter.setBrush(Qt.NoBrush)
        if self.object_name == "garrafa":
            self._draw_bottle(painter, caixa)
        else:
            self._draw_ball(painter, caixa)
        
        # Home position
        raio_home = max(16.0, min(width, height) * 0.028)
        painter.setPen(QPen(self.DIM, esp * 1.2))
        painter.drawEllipse(QPointF(home_x, home_y), raio_home, raio_home)
        
        # Labels
        fonte = QFont("Arial", max(12, int(min(width, height) * 0.026)), QFont.Bold)
        painter.setFont(fonte)
        painter.setPen(QPen(self.WHITE))
        mao = "DIREITA" if self.hand == "direita" else "ESQUERDA"
        painter.drawText(
            QRectF(home_x - width * 0.20, home_y + raio_home * 1.4,
                   width * 0.40, height * 0.08),
            Qt.AlignHCenter | Qt.AlignTop, f"MAO {mao} (home)")
        painter.drawText(
            QRectF(alvo_x - width * 0.22, alvo_y - raio * 2.4,
                   width * 0.44, height * 0.08),
            Qt.AlignHCenter | Qt.AlignBottom,
            f"{str(self.object_name).upper()} ({lado.upper()})")
        painter.setPen(QPen(self.DIM))
        painter.setFont(QFont("Arial",
                              max(10, int(fonte.pointSize() * 0.72)),
                              QFont.Bold))
        painter.drawText(QRectF(width * 0.10, height * 0.045,
                                width * 0.80, height * 0.07),
                         Qt.AlignHCenter | Qt.AlignVCenter,
                         "ANIMACAO MI: TRAJETORIA REAL DA ME")