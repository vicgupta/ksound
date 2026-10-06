package main

import (
	"fmt"

	"github.com/spf13/cobra"
)

func newDevicesCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list-devices",
		Short: "List available capture devices",
		RunE: func(cmd *cobra.Command, args []string) error {
			names, err := ListCaptureDevices()
			if err != nil {
				return err
			}
			for _, n := range names {
				fmt.Println("-", n)
			}
			return nil
		},
	}
}
