package main

import (
	"encoding/json"
	"fmt"

	"github.com/spf13/cobra"
)

func newDevicesCmd() *cobra.Command {
	var asJSON bool
	cmd := &cobra.Command{
		Use:   "list-devices",
		Short: "List available capture devices",
		RunE: func(cmd *cobra.Command, args []string) error {
			names, err := ListCaptureDevices()
			if err != nil {
				return err
			}
			if asJSON {
				enc := json.NewEncoder(cmd.OutOrStdout())
				enc.SetIndent("", "  ")
				return enc.Encode(names)
			}
			for _, n := range names {
				fmt.Fprintln(cmd.OutOrStdout(), "-", n)
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "output device list as JSON")
	return cmd
}
